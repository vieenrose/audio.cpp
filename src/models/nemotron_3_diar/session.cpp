#include "engine/models/nemotron_3_diar/session.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_3_diar {

// Diagnostic phase accounting for the diar encoder, off unless AUDIOCPP_PROF is set. audiocpp computes a
// PLANNED graph on a bare ggml backend (no ggml_backend_sched), so there is no eval-callback hook to borrow
// the way the x-asr leg has one; what is measurable here without distorting the run is phase-level time plus
// a structural op histogram. That is what a one-runtime port needs anyway: which phase the window's cost is
// actually in (mel frontend, conv subsample, transformer encoder, AOS state update), and which ops dominate
// the graph it would have to rebuild.
namespace diar_prof {

struct acc {
    double features_s = 0, pre_encode_s = 0, encode_s = 0, aos_s = 0, encode_build_s = 0;
    std::vector<double> window_s;   // per-window encode time: is the cost fixed or per-frame?
    std::vector<long> window_minflt;  // minor page faults per window: first touch of the compute arena
    std::vector<int64_t> window_frames;  // packed frames per window, for a per-frame view
    int64_t windows = 0, nodes = 0, elements = 0;
    std::map<int, int64_t> op_nodes, op_elems;
    // MUL_MAT shapes: the x-asr leg was measured at n = 3-24 columns, which is why it reaches single-digit
    // percent of int8 peak whatever the kernel. The question for the diar leg is whether it is the same
    // problem or the opposite one: its windows are 380-548 frames, so its GEMMs may already be fat. This is a
    // structural count (nodes and elements per shape), not a timing, because this graph is planned and run on
    // a bare ggml backend with no eval callback.
    std::map<std::pair<int64_t, int64_t>, int64_t> mm_shape_nodes, mm_shape_elems;
    bool on = false;
};

inline acc& get() {
    static acc a;
    return a;
}

inline bool on() {
    acc& a = get();
    if (!a.on) {
        a.on = std::getenv("AUDIOCPP_PROF") != nullptr;
    }
    return a.on;
}

inline void note_graph(ggml_cgraph * gf) {
    acc& a = get();
    if (!a.on || gf == nullptr) return;
    a.windows++;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        const int64_t ne = ggml_nelements(t);
        a.nodes++;
        a.elements += ne;
        a.op_nodes[(int) t->op] += 1;
        a.op_elems[(int) t->op] += ne;
        if (t->op == GGML_OP_MUL_MAT) {
            a.mm_shape_nodes[{t->ne[0], t->ne[1]}] += 1;
            a.mm_shape_elems[{t->ne[0], t->ne[1]}] += ne;
        }
    }
}

inline void dump() {
    acc& a = get();
    if (!a.on || a.windows == 0) return;
    std::fprintf(stderr,
                 "[diar-prof] windows=%lld graph_nodes/window=%.0f elems/window=%.0f | "
                 "features %.3f s  pre_encode %.3f s  encode %.3f s (build %.3f)  aos %.3f s\n",
                 (long long) a.windows, (double) a.nodes / a.windows, (double) a.elements / a.windows,
                 a.features_s, a.pre_encode_s, a.encode_s, a.encode_build_s, a.aos_s);
    std::fprintf(stderr, "[diar-prof] per-window encode s:");
    for (double w : a.window_s) std::fprintf(stderr, " %.3f", w);
    std::fprintf(stderr, "\n[diar-prof] per-window frames:");
    for (int64_t f : a.window_frames) std::fprintf(stderr, " %lld", (long long) f);
    std::fprintf(stderr, "  ms/frame:");
    for (size_t i = 0; i < a.window_s.size(); ++i)
        std::fprintf(stderr, " %.2f", a.window_s[i] * 1000.0 / (a.window_frames[i] ? a.window_frames[i] : 1));
    std::fprintf(stderr, "\n[diar-prof] per-window minor faults:");
    long prev = 0;
    for (size_t i = 0; i < a.window_minflt.size(); ++i) {
        std::fprintf(stderr, " %ld", a.window_minflt[i] - prev);
        prev = a.window_minflt[i];
    }
    std::fprintf(stderr, "   (a first window that is much more expensive AND fault-heavy is first touch of the\n"
                          "    compute arena, not arithmetic - arena bytes are the thing to shrink or prefault)\n");
    if (!a.mm_shape_nodes.empty()) {
        std::vector<std::pair<std::pair<int64_t, int64_t>, int64_t>> sh(a.mm_shape_nodes.begin(), a.mm_shape_nodes.end());
        std::sort(sh.begin(), sh.end(), [](auto& l, auto& r) { return l.second > r.second; });
        std::fprintf(stderr, "[diar-prof] MUL_MAT shapes (k x n, nodes/window, elems/window):\n");
        for (size_t i = 0; i < sh.size() && i < 10; ++i) {
            std::fprintf(stderr, "[diar-prof]   k=%-6lld n=%-6lld %8lld nodes  %12.0f elems\n",
                         (long long) sh[i].first.first, (long long) sh[i].first.second,
                         (long long)(sh[i].second / a.windows),
                         (double) a.mm_shape_elems[sh[i].first] / a.windows);
        }
        std::fprintf(stderr, "[diar-prof]   (%zu distinct shapes)\n", sh.size());
    }
    std::vector<std::pair<int, int64_t>> rows(a.op_nodes.begin(), a.op_nodes.end());
    std::sort(rows.begin(), rows.end(), [](auto& l, auto& r) { return l.second > r.second; });
    for (auto& r : rows) {
        const char* n = ggml_op_name((ggml_op) r.first);
        std::fprintf(stderr, "[diar-prof]   %-22s %7lld nodes/window  %10.0f elems/window\n",
                     n ? n : "?", (long long)(r.second / a.windows), (double) a.op_elems[r.first] / a.windows);
    }
    a = acc();
    a.on = true;
}

} // namespace diar_prof

namespace {

constexpr const char * kFamily = "nemotron_3_diar";
constexpr int64_t kSampleRate = 16000;
constexpr int64_t kOutputHopSamples = 160;
constexpr size_t kDefaultGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultWeightContextBytes = 1024ull * 1024ull * 1024ull;

template <typename T>
std::shared_ptr<const T> require_value(std::shared_ptr<const T> value, const char * label) {
    if (value == nullptr) throw std::runtime_error(std::string("Nemotron 3 diarization requires ") + label);
    return value;
}

std::string speaker_id(int64_t speaker) {
    return "speaker_" + std::to_string(speaker);
}

}  // namespace

Session::Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Assets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_value(std::move(assets), "assets")),
      contract_(require_value(std::move(contract), "model contract")),
      streaming_config_(streaming_profile(assets_->model_config.streaming, options.options)),
      graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"nemotron_3_diar.graph_arena_mb"}, kDefaultGraphArenaBytes)),
      weight_context_bytes_(runtime::parse_size_mb_option(
          options.options, {"nemotron_3_diar.weight_context_mb"}, kDefaultWeightContextBytes)) {
    if (task_.task != runtime::VoiceTaskKind::Diarization) {
        throw std::runtime_error("Nemotron 3 Diarization only supports diarization");
    }
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Nemotron 3 diarization");
    const auto storage = runtime::parse_tensor_storage_option(
        options.options,
        "nemotron_3_diar.weight_type",
        "nemotron_3_diar.weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    weights_ = load_weights(
        *assets_, execution_context().backend(), execution_context().backend_type(),
        storage, weight_context_bytes_);
    assets_->model_weights->release_storage();
    stream_scheduler_ = std::make_unique<StreamScheduler>(
        assets_->feature_config, streaming_config_, assets_->model_config.encoder.subsampling_factor);
}

Session::~Session() = default;

std::string Session::family() const { return kFamily; }
runtime::VoiceTaskKind Session::task_kind() const { return task_.task; }
runtime::RunMode Session::run_mode() const { return task_.mode; }

void Session::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.audio.has_value() && request.audio->sample_rate != 0 && request.audio->sample_rate != kSampleRate) {
        throw std::runtime_error("Nemotron 3 diarization requires 16 kHz audio");
    }
    mark_prepared();
    if (task_.mode == runtime::RunMode::Streaming) reset();
}

Session::DecodeConfig Session::decode_config(
    const std::unordered_map<std::string, std::string> & options) const {
    DecodeConfig config;
    if (const auto value = runtime::parse_finite_float_option(options, {"speaker_threshold"})) config.threshold = *value;
    if (const auto value = runtime::parse_i64_option(options, {"speaker_min_frames"})) config.min_frames = *value;
    if (const auto value = runtime::parse_i64_option(options, {"speaker_pad_frames"})) config.pad_frames = *value;
    if (config.threshold < 0.0F || config.threshold > 1.0F || config.min_frames < 0 || config.pad_frames < 0) {
        throw std::runtime_error("invalid Nemotron 3 diarization decode options");
    }
    return config;
}

std::vector<float> Session::pre_encode(const FeatureBatch & features) {
    const bool prof_p = diar_prof::on();
    const auto tp0 = prof_p ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ensure_pre_encode_graph(
        pre_encode_graph_, execution_context(), *assets_, *weights_, graph_arena_bytes_,
        features.batch, features.encoder_frames);
    core::write_tensor_f32(pre_encode_graph_->input, features.stacked);
    core::set_backend_threads(execution_context().backend(), pre_encode_graph_->threads);
    if (core::compute_backend_graph(
            execution_context().backend(), pre_encode_graph_->graph, pre_encode_graph_->plan) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron 3 diarization pre-encode graph failed");
    }
    std::vector<float> embeddings;
    core::read_tensor_f32_into(pre_encode_graph_->output.tensor, embeddings);
    if (prof_p) {
        diar_prof::get().pre_encode_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tp0).count();
    }
    return embeddings;
}

std::vector<float> Session::encode(
    const std::vector<float> & embeddings,
    int64_t batch,
    int64_t frames,
    const std::vector<int64_t> & valid_frames) {
    const bool prof = diar_prof::on();
    const auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ensure_encoder_graph(
        encoder_graph_, execution_context(), *assets_, *weights_, graph_arena_bytes_,
        batch, frames);
    core::write_tensor_f32(encoder_graph_->input, embeddings);
    core::write_tensor_f16(
        encoder_graph_->attention_mask,
        attention_mask(valid_frames, frames));
    core::set_backend_threads(execution_context().backend(), encoder_graph_->threads);
    if (prof) {
        diar_prof::note_graph(encoder_graph_->graph);
        diar_prof::get().encode_build_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }
    const auto t_compute0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (core::compute_backend_graph(
            execution_context().backend(), encoder_graph_->graph, encoder_graph_->plan) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron 3 diarization encoder graph failed");
    }
    if (const char * dir = std::getenv("AUDIOCPP_DUMP_LAYER0"); dir != nullptr && encoder_graph_->dump_layer0) {
        // Ground truth for the one-runtime port: audio.cpp's OWN layer 0, input and output, so a layer
        // rebuilt in CrispASR's ggml can be diffed against it byte for byte. Raw F32 plus a .shape line
        // with ggml's ne0 (hidden) and ne1 (frames), which is all a port needs to reproduce it.
        const std::string base = std::string(dir) + "/layer0";
        std::vector<float> tin(ggml_nelements(encoder_graph_->layer0_in.tensor));
        core::read_tensor_f32_into(encoder_graph_->layer0_in.tensor, tin);
        std::vector<float> tout(ggml_nelements(encoder_graph_->layer0_out.tensor));
        core::read_tensor_f32_into(encoder_graph_->layer0_out.tensor, tout);
        auto dump = [&](const char * suffix, const std::vector<float> & v, bool shape) {
            std::FILE * fh = std::fopen((base + suffix).c_str(), "wb");
            if (fh == nullptr) return;
            if (shape) {
                std::fprintf(fh, "%lld %lld\n", (long long) encoder_graph_->layer0_in.tensor->ne[0],
                             (long long) encoder_graph_->layer0_in.tensor->ne[1]);
            } else {
                std::fwrite(v.data(), sizeof(float), v.size(), fh);
            }
            std::fclose(fh);
        };
        dump("_in.f32", tin, false);
        dump("_in.shape", tin, true);
        dump("_out.f32", tout, false);
        std::fprintf(stderr, "[diar-dump] layer0 in %zu floats, out %zu floats (hidden=%lld frames=%lld) -> %s\n",
                     tin.size(), tout.size(), (long long) encoder_graph_->layer0_in.tensor->ne[0],
                     (long long) encoder_graph_->layer0_in.tensor->ne[1], dir);
    }
    if (prof) {
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        diar_prof::get().encode_s += dt;
        diar_prof::get().window_s.push_back(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_compute0).count());
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        diar_prof::get().window_minflt.push_back(ru.ru_minflt);
        diar_prof::get().window_frames.push_back(frames);
    }
    std::vector<float> probabilities;
    core::read_tensor_f32_into(encoder_graph_->probabilities.tensor, probabilities);
    return probabilities;
}

void Session::process_window_batch(
    const std::vector<StreamWindow> & windows,
    const std::vector<AoscState *> & states,
    const std::vector<std::vector<float> *> & timelines) {
    if (windows.empty()) return;
    if (windows.size() != states.size() || windows.size() != timelines.size()) {
        throw std::runtime_error("Nemotron 3 diarization streaming batch shape mismatch");
    }
    const bool prof_w = diar_prof::on();
    const auto tf0 = prof_w ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto features = compute_stream_features(
        windows, *assets_, execution_context().config().threads);
    if (prof_w) {
        diar_prof::get().features_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tf0).count();
    }
    if (task_.mode == runtime::RunMode::Streaming) {
        if (windows.size() != 1) {
            throw std::runtime_error("Nemotron 3 diarization streaming requires a single graph row");
        }
        const int64_t capacity = streaming_config_.chunk_left_context +
            streaming_config_.chunk_len + streaming_config_.chunk_right_context;
        if (features.encoder_frames > capacity) {
            throw std::runtime_error("Nemotron 3 diarization streaming window exceeds graph capacity");
        }
        // Do NOT pad the window up to the configured capacity. encode() takes the length as a graph
        // dimension (ensure_encoder_graph rebuilds when it changes) and the encoder is already given an
        // explicit attention_mask(valid_frames, frames), so the zero-padded tail contributed no
        // information - it contributed encoder FLOPs. The drain window is the pathological case: a couple
        // of seconds of audio at the end of a file was costing a full-capacity encoder pass, which is where
        // this loop's unattributed "other" seconds came from. Steady-state windows are already at capacity
        // (the engine only releases a window once chunk_len frames have accumulated), so this trims the
        // start and the drain and leaves the steady state on the same graph shape.
        if (features.encoder_frames <= 0) {
            throw std::runtime_error("Nemotron 3 diarization streaming window is empty");
        }
    }
    const auto chunk_embeddings = pre_encode(features);
    std::vector<int64_t> central_frames;
    std::vector<int64_t> left_context_frames;
    std::vector<int64_t> right_context_frames;
    central_frames.reserve(windows.size());
    left_context_frames.reserve(windows.size());
    right_context_frames.reserve(windows.size());
    for (size_t row = 0; row < windows.size(); ++row) {
        central_frames.push_back(windows[row].encoder_frames);
        left_context_frames.push_back(windows[row].left_context_frames);
        right_context_frames.push_back(windows[row].right_context_frames);
    }
    process_embedding_batch(
        chunk_embeddings, static_cast<int64_t>(windows.size()), features.encoder_frames,
        features.valid_encoder_frames, central_frames, left_context_frames,
        right_context_frames, states, timelines);
}

void Session::process_embedding_batch(
    const std::vector<float> & chunk_embeddings,
    int64_t batch,
    int64_t chunk_capacity,
    const std::vector<int64_t> & chunk_frames,
    const std::vector<int64_t> & central_frames,
    const std::vector<int64_t> & left_context_frames,
    const std::vector<int64_t> & right_context_frames,
    const std::vector<AoscState *> & states,
    const std::vector<std::vector<float> *> & timelines) {
    const size_t rows = static_cast<size_t>(batch);
    if (batch <= 0 || chunk_frames.size() != rows || central_frames.size() != rows ||
        left_context_frames.size() != rows || right_context_frames.size() != rows ||
        states.size() != rows || timelines.size() != rows) {
        throw std::runtime_error("Nemotron 3 diarization embedding batch shape mismatch");
    }
    const int64_t hidden = assets_->model_config.encoder.hidden_size;
    const int64_t speakers = assets_->model_config.num_speakers;
    const int64_t upsample = assets_->model_config.head.upsample_factor;
    if (chunk_embeddings.size() != rows * static_cast<size_t>(chunk_capacity * hidden)) {
        throw std::runtime_error("Nemotron 3 diarization embedding batch data mismatch");
    }

    std::vector<int64_t> packed_lengths;
    packed_lengths.reserve(rows);
    int64_t packed_capacity = 0;
    for (size_t row = 0; row < rows; ++row) {
        const int64_t state_frames = states[row]->speaker_cache_frames() + states[row]->fifo_frames();
        packed_lengths.push_back(state_frames + chunk_frames[row]);
        packed_capacity = std::max(packed_capacity, state_frames + chunk_frames[row]);
    }
    if (task_.mode == runtime::RunMode::Streaming) {
        const int64_t capacity = streaming_config_.spkcache_len + streaming_config_.fifo_len + chunk_capacity;
        if (packed_capacity > capacity) {
            throw std::runtime_error("Nemotron 3 diarization streaming state exceeds graph capacity");
        }
        // packed_capacity stays at the real max(packed_lengths) instead of being forced up to the
        // configured capacity. The rows are masked by packed_lengths in attention_mask() and the tail is
        // zero-filled, so a shorter buffer is the same computation on fewer positions - and for the drain
        // window that is most of the pass. Same reason as the window padding above: the padding was
        // capacity-shaped work, not information.
    }
    std::vector<float> packed(
        rows * static_cast<size_t>(packed_capacity * hidden), 0.0F);
    for (size_t row = 0; row < rows; ++row) {
        auto * output = packed.data() + row * static_cast<size_t>(packed_capacity * hidden);
        const auto & cache = states[row]->speaker_cache();
        const auto & fifo = states[row]->fifo();
        std::copy(cache.begin(), cache.end(), output);
        std::copy(fifo.begin(), fifo.end(), output + cache.size());
        const auto * chunk = chunk_embeddings.data() + row * static_cast<size_t>(chunk_capacity * hidden);
        std::copy_n(
            chunk, static_cast<size_t>(chunk_frames[row] * hidden),
            output + cache.size() + fifo.size());
    }

    const auto high_resolution = encode(
        packed, batch, packed_capacity, packed_lengths);
    const int64_t high_capacity = packed_capacity * upsample;
    for (size_t row = 0; row < rows; ++row) {
        const int64_t old_cache = states[row]->speaker_cache_frames();
        const int64_t old_fifo = states[row]->fifo_frames();
        const int64_t valid_chunk_frames = chunk_frames[row];
        const int64_t left = std::min<int64_t>(left_context_frames[row], valid_chunk_frames);
        const int64_t right = std::min<int64_t>(
            right_context_frames[row], valid_chunk_frames - left);
        const int64_t central = std::min<int64_t>(
            central_frames[row], valid_chunk_frames - left - right);
        if (central <= 0) continue;
        const auto * row_high = high_resolution.data() + row * static_cast<size_t>(high_capacity * speakers);
        const int64_t high_start = (old_cache + old_fifo + left) * upsample;
        timelines[row]->insert(
            timelines[row]->end(),
            row_high + static_cast<size_t>(high_start * speakers),
            row_high + static_cast<size_t>((high_start + central * upsample) * speakers));

        std::vector<float> low_resolution(static_cast<size_t>(packed_lengths[row] * speakers), 0.0F);
        for (int64_t frame = 0; frame < packed_lengths[row]; ++frame) {
            for (int64_t sub = 0; sub < upsample; ++sub) {
                const float * source = row_high + static_cast<size_t>((frame * upsample + sub) * speakers);
                float * destination = low_resolution.data() + static_cast<size_t>(frame * speakers);
                for (int64_t speaker = 0; speaker < speakers; ++speaker) {
                    destination[speaker] += source[speaker] / static_cast<float>(upsample);
                }
            }
        }
        const auto * row_chunk = chunk_embeddings.data() + row * static_cast<size_t>(chunk_capacity * hidden);
        states[row]->update(
            row_chunk, static_cast<int>(valid_chunk_frames), low_resolution.data(),
            static_cast<int>(left), static_cast<int>(right));
    }
}

std::vector<runtime::SpeakerTurn> Session::decode_turns(
    const std::vector<float> & probabilities,
    int64_t frames,
    const DecodeConfig & config,
    bool include_open_turns) const {
    const int64_t speakers = assets_->model_config.num_speakers;
    std::vector<runtime::SpeakerTurn> turns;
    for (int64_t speaker = 0; speaker < speakers; ++speaker) {
        int64_t start = -1;
        for (int64_t frame = 0; frame <= frames; ++frame) {
            const bool active = frame < frames &&
                probabilities[static_cast<size_t>(frame * speakers + speaker)] >= config.threshold;
            if (active && start < 0) {
                start = frame;
            } else if (!active && start >= 0) {
                const int64_t end = frame;
                if (include_open_turns || end < frames) {
                    const int64_t padded_start = std::max<int64_t>(0, start - config.pad_frames);
                    const int64_t padded_end = std::min<int64_t>(frames, end + config.pad_frames);
                    if (padded_end - padded_start >= config.min_frames) {
                        float confidence = 0.0F;
                        for (int64_t index = start; index < end; ++index) {
                            confidence += probabilities[static_cast<size_t>(index * speakers + speaker)];
                        }
                        confidence /= static_cast<float>(std::max<int64_t>(1, end - start));
                        turns.push_back({
                            {padded_start * kOutputHopSamples, padded_end * kOutputHopSamples},
                            speaker_id(speaker), confidence, "",
                        });
                    }
                }
                start = -1;
            }
        }
    }
    std::sort(turns.begin(), turns.end(), [](const auto & left, const auto & right) {
        if (left.span.start_sample != right.span.start_sample) return left.span.start_sample < right.span.start_sample;
        return left.speaker_id < right.speaker_id;
    });
    return turns;
}

runtime::TaskResult Session::run(const runtime::TaskRequest & request) {
    auto results = run_batch({request});
    return std::move(results.front());
}

std::vector<runtime::TaskResult> Session::run_batch(
    const std::vector<runtime::TaskRequest> & requests) {
    std::vector<runtime::TaskResult> results(requests.size());
    run_batch(requests, [&](size_t index, runtime::TaskResult result) {
        results[index] = std::move(result);
    });
    return results;
}

void Session::run_batch(
    const std::vector<runtime::TaskRequest> & requests,
    const runtime::IBatchedOfflineVoiceTaskSession::ResultCallback & on_result) {
    require_prepared("Nemotron 3 diarization run_batch()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Nemotron 3 diarization run_batch() requires offline mode");
    }
    if (requests.empty()) throw std::runtime_error("Nemotron 3 diarization batch must not be empty");
    std::vector<DecodeConfig> decoding;
    decoding.reserve(requests.size());
    std::vector<std::unique_ptr<AoscState>> states;
    std::vector<std::vector<float>> probabilities(requests.size());
    states.reserve(requests.size());
    std::vector<runtime::AudioBuffer> audio;
    audio.reserve(requests.size());
    for (const auto & request : requests) {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Nemotron 3 diarization");
        if (!request.audio_input.has_value() || request.audio_input->sample_rate != kSampleRate ||
            request.audio_input->channels <= 0) {
            throw std::runtime_error("Nemotron 3 diarization requires 16 kHz audio input");
        }
        decoding.push_back(decode_config(request.options));
        audio.push_back(*request.audio_input);
        states.push_back(std::make_unique<AoscState>(
            streaming_config_, static_cast<int>(assets_->model_config.num_speakers),
            static_cast<int>(assets_->model_config.encoder.hidden_size), assets_->silence_embedding));
    }

    const auto features = compute_features(audio, *assets_, execution_context().config().threads);
    const auto full_embeddings = pre_encode(features);
    const int64_t hidden = assets_->model_config.encoder.hidden_size;
    const int64_t chunk_len = streaming_config_.chunk_len;
    const int64_t speakers = assets_->model_config.num_speakers;
    std::vector<bool> completed(requests.size(), false);
    auto finish_row = [&](size_t row) {
        const int64_t input_samples = static_cast<int64_t>(
            requests[row].audio_input->samples.size() /
            static_cast<size_t>(requests[row].audio_input->channels));
        const int64_t expected_frames = (input_samples + kOutputHopSamples - 1) / kOutputHopSamples;
        const int64_t frames = std::min<int64_t>(
            expected_frames, static_cast<int64_t>(probabilities[row].size() / speakers));
        probabilities[row].resize(static_cast<size_t>(frames * speakers));
        runtime::TaskResult result;
        result.speaker_turns = decode_turns(probabilities[row], frames, decoding[row], true);
        completed[row] = true;
        on_result(row, std::move(result));
    };
    for (int64_t start = 0;; start += chunk_len) {
        bool any_active = false;
        int64_t chunk_capacity = 0;
        for (size_t row = 0; row < requests.size(); ++row) {
            if (start >= features.valid_encoder_frames[row]) continue;
            any_active = true;
            const int64_t left = std::min<int64_t>(streaming_config_.chunk_left_context, start);
            const int64_t end = std::min<int64_t>(start + chunk_len, features.valid_encoder_frames[row]);
            const int64_t right = std::min<int64_t>(
                streaming_config_.chunk_right_context, features.valid_encoder_frames[row] - end);
            chunk_capacity = std::max(chunk_capacity, left + end - start + right);
        }
        if (!any_active) break;

        std::vector<float> chunk_embeddings(
            requests.size() * static_cast<size_t>(chunk_capacity * hidden), 0.0F);
        std::vector<int64_t> chunk_frames;
        std::vector<int64_t> central_frames;
        std::vector<int64_t> left_context_frames;
        std::vector<int64_t> right_context_frames;
        std::vector<AoscState *> active_states;
        std::vector<std::vector<float> *> timelines;
        for (size_t row = 0; row < requests.size(); ++row) {
            active_states.push_back(states[row].get());
            timelines.push_back(&probabilities[row]);
            if (start >= features.valid_encoder_frames[row]) {
                chunk_frames.push_back(0);
                central_frames.push_back(0);
                left_context_frames.push_back(0);
                right_context_frames.push_back(0);
                continue;
            }
            const int64_t left = std::min<int64_t>(streaming_config_.chunk_left_context, start);
            const int64_t end = std::min<int64_t>(start + chunk_len, features.valid_encoder_frames[row]);
            const int64_t right = std::min<int64_t>(
                streaming_config_.chunk_right_context, features.valid_encoder_frames[row] - end);
            const int64_t frames = left + end - start + right;
            const auto * source = full_embeddings.data() +
                (row * static_cast<size_t>(features.encoder_frames) + static_cast<size_t>(start - left)) * hidden;
            auto * destination = chunk_embeddings.data() + row * static_cast<size_t>(chunk_capacity * hidden);
            std::copy_n(source, static_cast<size_t>(frames * hidden), destination);
            chunk_frames.push_back(frames);
            central_frames.push_back(end - start);
            left_context_frames.push_back(left);
            right_context_frames.push_back(right);
        }
        process_embedding_batch(
            chunk_embeddings, static_cast<int64_t>(requests.size()), chunk_capacity,
            chunk_frames, central_frames, left_context_frames, right_context_frames,
            active_states, timelines);
        for (size_t row = 0; row < requests.size(); ++row) {
            if (!completed[row] && start + chunk_len >= features.valid_encoder_frames[row]) {
                finish_row(row);
            }
        }
    }
    for (size_t row = 0; row < requests.size(); ++row) {
        if (!completed[row]) {
            finish_row(row);
        }
    }
}

runtime::StreamingPolicy Session::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    policy.preferred_audio_chunk_samples = streaming_config_.chunk_len * 1280;
    policy.preferred_audio_chunk_seconds =
        static_cast<double>(policy.preferred_audio_chunk_samples) / kSampleRate;
    return policy;
}

void Session::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Nemotron 3 diarization start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron 3 diarization start_stream() requires streaming mode");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Nemotron 3 diarization");
    reset();
    stream_request_ = request;
    stream_request_.audio_input.reset();
    stream_started_ = true;
}

void Session::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void Session::reset() {
    stream_scheduler_->reset();
    stream_state_ = std::make_unique<AoscState>(
        streaming_config_, static_cast<int>(assets_->model_config.num_speakers),
        static_cast<int>(assets_->model_config.encoder.hidden_size), assets_->silence_embedding);
    stream_probabilities_.clear();
    stream_samples_ = 0;
    emitted_stream_turns_.clear();
    stream_request_ = {};
    stream_started_ = false;
}

runtime::StreamEvent Session::process_stream_windows(const std::vector<StreamWindow> & windows) {
    for (const auto & window : windows) {
        process_window_batch({window}, {stream_state_.get()}, {&stream_probabilities_});
    }
    const int64_t frames = static_cast<int64_t>(
        stream_probabilities_.size() / assets_->model_config.num_speakers);
    const auto turns = decode_turns(
        stream_probabilities_, frames, decode_config(stream_request_.options), false);
    runtime::StreamEvent event;
    for (const auto & turn : turns) {
        const auto key = std::make_tuple(
            turn.span.start_sample, turn.span.end_sample, turn.speaker_id);
        if (emitted_stream_turns_.insert(key).second) {
            event.speaker_turns.push_back(turn);
        }
    }
    if (!event.speaker_turns.empty() && stream_event_sink_) {
        stream_event_sink_(event);
        return {};
    }
    return event;
}

runtime::StreamEvent Session::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Nemotron 3 diarization process_audio_chunk()");
    if (!stream_started_ || chunk.sample_rate != kSampleRate || chunk.channels != 1) {
        throw std::runtime_error("Nemotron 3 diarization received an invalid streaming audio chunk");
    }
    stream_samples_ += static_cast<int64_t>(chunk.samples.size());
    return process_stream_windows(stream_scheduler_->push_audio(chunk));
}

runtime::TaskResult Session::finalize() {
    require_prepared("Nemotron 3 diarization finalize()");
    if (!stream_started_) throw std::runtime_error("Nemotron 3 diarization has no active stream");
    // Diagnostic split of the drain, which shows up as unattributed "other" seconds in the composite loop
    // (the per-piece timers only cover audiocpp_stream_push). Off unless AUDIOCPP_PROF is set.
    const bool prof = std::getenv("AUDIOCPP_PROF") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    process_stream_windows(stream_scheduler_->finalize());
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t speakers = assets_->model_config.num_speakers;
    const int64_t expected_frames = (stream_samples_ + kOutputHopSamples - 1) / kOutputHopSamples;
    const int64_t frames = std::min<int64_t>(
        expected_frames, static_cast<int64_t>(stream_probabilities_.size() / speakers));
    stream_probabilities_.resize(static_cast<size_t>(frames * speakers));
    runtime::TaskResult result;
    result.speaker_turns = decode_turns(
        stream_probabilities_, frames, decode_config(stream_request_.options), true);
    const auto t2 = std::chrono::steady_clock::now();
    if (prof) {
        std::fprintf(stderr, "[audiocpp-prof] finalize: final_window %.3f s  decode_turns %.3f s  (frames=%lld)\n",
                     std::chrono::duration<double>(t1 - t0).count(),
                     std::chrono::duration<double>(t2 - t1).count(), (long long)frames);
    }
    if (prof) {
        diar_prof::get().aos_s += std::chrono::duration<double>(t2 - t1).count();
        diar_prof::dump();
    }
    stream_started_ = false;
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_3_diar_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = kFamily;
    config.load_assets = [](const std::filesystem::path & model_path) {
        return load_assets(model_path, model_spec::default_package_spec_path(kFamily));
    };
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const Assets> assets,
                                std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::unique_ptr<runtime::IVoiceTaskSession>(
            std::make_unique<Session>(task, options, std::move(assets), std::move(contract)));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::nemotron_3_diar

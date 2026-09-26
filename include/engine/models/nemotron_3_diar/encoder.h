#pragma once

#include "engine/models/nemotron_3_diar/assets.h"
#include "engine/framework/core/execution_context.h"
#include "ggml-alloc.h"

#include <memory>
#include <vector>

namespace engine::models::nemotron_3_diar {

struct PreEncodeGraph {
    int64_t batch = 0;
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_gallocr_t allocator = nullptr;
    int threads = 1;
    core::TensorValue input;
    core::TensorValue output;
    ~PreEncodeGraph();
};

struct EncoderGraph {
    int64_t batch = 0;
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_gallocr_t allocator = nullptr;
    int threads = 1;
    core::TensorValue input;
    core::TensorValue attention_mask;
    core::TensorValue rope_cos;
    core::TensorValue rope_sin;
    core::TensorValue probabilities;
    // Ground-truth handles for the one-runtime port (docs/one-runtime-merge.md, stage 2). Layer 0's input
    // and output are captured as graph outputs ONLY when AUDIOCPP_DUMP_LAYER0 names a directory, so the
    // default path is unchanged and the check costs one getenv at graph-build time.
    core::TensorValue layer0_in;
    core::TensorValue layer0_out;
    // The full encoder stack's real output (after all `weights.layers.size()` layers, before final_norm and
    // the head) - the acceptance test for porting the WHOLE stack, not just one layer. Same dump_layer0 gate.
    core::TensorValue encoder_out;
    bool dump_layer0 = false;
    ~EncoderGraph();
};

// Named intermediate tensors of the traced encoder layer (AUDIOCPP_TRACE_LAYER0), for the one-runtime port.
struct TracedStage {
    std::string name;
    ggml_tensor * tensor = nullptr;
};
const std::vector<TracedStage> & traced_layer_stages();
void clear_traced_layer_stages();

// docs/one-runtime-merge.md SS12: the running 32-layer graph cannot be a sound oracle for its own
// intermediates - ggml_set_output on a per-layer stage does not stop a LATER, structurally identical layer's
// tensor from landing on the same gallocr slot (confirmed: 02_qkv, 05_v, 07_oproj and 11_ffn_out all read
// back corrupted despite the flag). This builds ONLY encoder layer 0 as its own graph with nothing else that
// could ever alias across LAYERS, which is enough to make the final output and most stages (01/03/04/06/08/10
// - verified byte-identical to the production trace) sound. It is NOT a complete fix: 07_oproj and 11_ffn_out
// still alias EACH OTHER within this single-layer graph (same bytes, ggml_set_output notwithstanding), and
// 02_qkv/05_v/09_ffn_in read back plausible but as yet uncross-checked values - see SS12 for the open
// question and use the true final output (`output` below) as the primary acceptance test. No production code
// path calls this - it exists for the port comparison only.
struct Layer0IsolatedResult {
    std::vector<float> output;          // [hidden, frames] - the layer's true final output
    std::vector<TracedStage> stages;    // named intermediates; valid until this result is destroyed
    ggml_context * ggml = nullptr;      // owns the tensor structs above
    ggml_gallocr_t allocator = nullptr; // owns the tensors' DATA - freeing it before `stages` is read back is
                                        // a real bug this function's first draft had: read_tensor_f32_into
                                        // silently returns zeros off a freed buffer, no error. Keep it alive
                                        // until the caller is done, then let this destructor free it.
    ~Layer0IsolatedResult();
    Layer0IsolatedResult() = default;
    Layer0IsolatedResult(Layer0IsolatedResult &&) noexcept;
    Layer0IsolatedResult & operator=(Layer0IsolatedResult &&) noexcept;
    Layer0IsolatedResult(const Layer0IsolatedResult &) = delete;
    Layer0IsolatedResult & operator=(const Layer0IsolatedResult &) = delete;
};

Layer0IsolatedResult run_layer0_isolated(
    ggml_backend_t backend,
    const EncoderLayerWeights & weights,
    const EncoderConfig & config,
    const std::vector<float> & input,   // [hidden, frames], hidden fastest - same layout as layer0_in.f32
    int64_t frames,
    int64_t valid_frames);              // <= frames; matches attention_mask()'s [valid_frames, frames) pad

// Same idea, for the WHOLE encoder stack (all `weights.layers.size()` layers, chained in one graph) rather
// than one layer - the acceptance test for porting the entire encoder, not just a single layer. Only the
// final output is needed here (no per-layer stage tracing), so ownership can stay entirely inside the
// function: everything is read back before it returns.
std::vector<float> run_encoder_isolated(
    ggml_backend_t backend,
    const ModelWeights & weights,
    const EncoderConfig & config,
    const std::vector<float> & input,   // [hidden, frames], hidden fastest - same layout as encoder_in.f32
    int64_t frames,
    int64_t valid_frames);

// Same idea for the "head" that turns the encoder's output into per-frame speaker probabilities
// (final_norm -> encoder_projection -> subpixel_upsample conv -> relu -> head_hidden -> relu -> speaker_head
// -> sigmoid; encoder.cpp's build_encoder_graph, the lines after the layer loop). Takes the encoder's real
// output directly (e.g. encoder_out.f32 / run_encoder_isolated's own return) rather than recomputing it.
std::vector<float> run_head_isolated(
    ggml_backend_t backend,
    const ModelWeights & weights,
    const ModelConfig & config,
    const std::vector<float> & encoder_output,   // [hidden, frames], hidden fastest
    int64_t frames);

void ensure_pre_encode_graph(
    std::unique_ptr<PreEncodeGraph> & graph,
    const core::ExecutionContext & execution,
    const Assets & assets,
    const ModelWeights & weights,
    size_t arena_bytes,
    int64_t batch,
    int64_t frames);

void ensure_encoder_graph(
    std::unique_ptr<EncoderGraph> & graph,
    const core::ExecutionContext & execution,
    const Assets & assets,
    const ModelWeights & weights,
    size_t arena_bytes,
    int64_t batch,
    int64_t frames);

std::vector<float> rope_table(int64_t batch, int64_t heads, int64_t frames, int64_t head_dim, float theta, bool cosine);
std::vector<float> attention_mask(const std::vector<int64_t> & lengths, int64_t frames);

}  // namespace engine::models::nemotron_3_diar

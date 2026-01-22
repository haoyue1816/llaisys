#include "qwen2_model.hpp"
#include "llaisys/models/qwen2.h"

__C __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    llaisys::models::Qwen2Model *model = nullptr;
    if (ndevice > 0) {
        model = new llaisys::models::Qwen2Model(*meta, device, device_ids[0]);
    } else {
        model = new llaisys::models::Qwen2Model(*meta, device, 0);
    }

    return reinterpret_cast<LlaisysQwen2Model*>(model);
}

__C __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        delete reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    }
}

__C __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (model) {
        llaisys::models::Qwen2Model *m = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        return m->getWeights();
    }
    return nullptr;
}

__C __export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {

    if (model) {
        llaisys::models::Qwen2Model *m = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        return m->infer(token_ids, ntoken);
    }
    return -1;
}

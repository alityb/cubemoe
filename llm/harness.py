"""Routing capture + expert knockout for Hugging Face MoE LMs (Granite-MoE now; OLMoE added later).
Each MoE router gets: ._record (list to append top-k indices to, or None) and ._knockout (set of expert
ids whose router logits are set to -inf before top-k). Both off => the original computation, unchanged."""
import types, torch, torch.nn.functional as F
def _granite_router_forward(self, hidden_states):
    logits = F.linear(hidden_states, self.weight).float()
    if self._knockout:
        logits = logits.clone(); logits[:, list(self._knockout)] = float('-inf')
    top_k_logits, top_k_index = logits.topk(self.top_k, dim=-1)
    if self._record is not None: self._record.append(top_k_index.detach().cpu())
    return top_k_index, torch.softmax(top_k_logits, dim=-1).type_as(hidden_states), logits
def routers(model):
    out=[]
    for L,layer in enumerate(model.model.layers):
        r=layer.block_sparse_moe.router
        if not hasattr(r,'_knockout'):
            r._knockout=set(); r._record=None; r.forward=types.MethodType(_granite_router_forward, r)
        out.append(r)
    return out
def moe_blocks(model): return [layer.block_sparse_moe for layer in model.model.layers]
def reset(model):
    for r in routers(model): r._knockout=set(); r._record=None

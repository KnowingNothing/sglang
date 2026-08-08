import torch


def compact_mori_routes(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    global_topk_ids: torch.Tensor,
    *,
    expert_offset: int,
    num_local_experts: int,
):
    """Turn MORI's received global routes into compact local top-k=1 rows."""

    local_topk_ids = global_topk_ids.to(torch.int64) - expert_offset
    valid = (local_topk_ids >= 0) & (local_topk_ids < num_local_experts)
    local_topk_ids = torch.where(
        valid, local_topk_ids, torch.full_like(local_topk_ids, -1)
    ).to(torch.int32)

    valid_positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    token_indices = torch.div(
        valid_positions, global_topk_ids.shape[1], rounding_mode="floor"
    )

    compact_hidden_states = hidden_states.index_select(0, token_indices)
    compact_topk_ids = (
        local_topk_ids.reshape(-1).index_select(0, valid_positions).reshape(-1, 1)
    )
    compact_topk_weights = torch.ones(
        (valid_positions.numel(), 1),
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )

    output_index = torch.full_like(local_topk_ids, -1, dtype=torch.int32)
    if valid_positions.numel() > 0:
        output_index.reshape(-1).index_copy_(
            0,
            valid_positions,
            torch.arange(
                valid_positions.numel(),
                dtype=torch.int32,
                device=valid_positions.device,
            ),
        )

    return (
        compact_hidden_states,
        compact_topk_weights,
        compact_topk_ids,
        local_topk_ids,
        output_index,
    )

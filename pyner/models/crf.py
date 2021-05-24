import torch

from pyner.torch_utils import masked_flip

IMPOSSIBLE = -100000


class LinearChainCRF(torch.nn.Module):
    def __init__(self, forbidden_transitions, start_forbidden_transitions=None, end_forbidden_transitions=None, learnable_transitions=True, with_start_end_transitions=True):
        super().__init__()

        num_tags = forbidden_transitions.shape[0]

        self.register_buffer('forbidden_transitions', forbidden_transitions.float())
        if start_forbidden_transitions is not None:
            self.register_buffer('start_forbidden_transitions', start_forbidden_transitions.float())
        else:
            self.register_buffer('start_forbidden_transitions', torch.zeros(num_tags, dtype=torch.float))
        if end_forbidden_transitions is not None:
            self.register_buffer('end_forbidden_transitions', end_forbidden_transitions.float())
        else:
            self.register_buffer('end_forbidden_transitions', torch.zeros(num_tags, dtype=torch.float))

        if learnable_transitions:
            self.transitions = torch.nn.Parameter(torch.zeros_like(forbidden_transitions, dtype=torch.float))
        else:
            self.register_buffer('transitions', torch.zeros_like(forbidden_transitions, dtype=torch.float))

        if learnable_transitions and with_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.zeros(num_tags, dtype=torch.float))
        else:
            self.register_buffer('start_transitions', torch.zeros(num_tags, dtype=torch.float))

        if learnable_transitions and with_start_end_transitions:
            self.end_transitions = torch.nn.Parameter(torch.zeros(num_tags, dtype=torch.float))
        else:
            self.register_buffer('end_transitions', torch.zeros(num_tags, dtype=torch.float))

    def propagate(self, emissions, mask, tags=None, ring_op_name="logsumexp", use_constraints=True, way="forward"):
        """
        Each alpha is the potential for the state given all previous observations and the current one
        """
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

        if tags is not None:
            if len(tags.shape) == 2:
                tags = tags.transpose(0, 1).unsqueeze(1)
            elif len(tags.shape) == 3:
                tags = tags.permute(2, 0, 1)

        backtrack = None

        if ring_op_name == "logsumexp":
            def ring_op(last_potential, trans, loc):
                return (last_potential.unsqueeze(-1) + trans.unsqueeze(0).unsqueeze(0)).logsumexp(2)
        elif ring_op_name == "posterior":
            def ring_op(last_potential, trans, loc):
                return trans[tags[loc]] + last_potential[torch.arange(tags.shape[1]).unsqueeze(1),
                                                         torch.arange(tags.shape[2]).unsqueeze(0),
                                                         tags[loc]].unsqueeze(-1)
        elif ring_op_name == "max":
            backtrack = []

            def ring_op(last_potential, trans, loc):
                res, indices = (last_potential.unsqueeze(-1) + trans.unsqueeze(0).unsqueeze(0)).max(2)
                backtrack.append(indices)
                return res
        else:
            raise NotImplementedError()

        if use_constraints:
            start_transitions = self.start_transitions.masked_fill(self.start_forbidden_transitions.bool(), IMPOSSIBLE)
            transitions = self.transitions.masked_fill(self.forbidden_transitions.bool(), IMPOSSIBLE)
            end_transitions = self.end_transitions.masked_fill(self.end_forbidden_transitions.bool(), IMPOSSIBLE)
        else:
            start_transitions = self.start_transitions
            transitions = self.transitions
            end_transitions = self.end_transitions

        if way == "backward":
            assert ring_op_name != "max", "Unsupported"
            start_transitions, end_transitions = end_transitions, start_transitions
            transitions = transitions.t()
            emissions = masked_flip(emissions.transpose(0, 1), mask.transpose(0, 1), -2).transpose(0, 1)
            # emissions = torch.cat([torch.zeros_like(emissions[[0]]), emissions[:-1]])
            # log_probs = [start_transitions.unsqueeze(0).unsqueeze(0).repeat_interleave(tags.shape[1] if tags is not None else 1, dim=0)]

        log_probs = [(start_transitions + emissions[0]).unsqueeze(0).repeat_interleave(tags.shape[1] if tags is not None else 1, dim=0)]

        for k in range(1, len(emissions)):
            res = ring_op(log_probs[-1], transitions, k - 1)
            log_probs.append(torch.where(
                mask[k].unsqueeze(-1),
                res + emissions[k],
                log_probs[-1]
            ))

        z = ring_op(log_probs[-1], end_transitions.unsqueeze(1),
                    ((mask.sum(0) - 1).unsqueeze(0), torch.arange(log_probs[-1].shape[0]).unsqueeze(1), torch.arange(mask.shape[1]).unsqueeze(0))).squeeze(-1)

        log_probs = torch.cat(log_probs, dim=0)

        if way == "backward":
            log_probs = masked_flip(
                log_probs.transpose(0, 1),
                mask.transpose(0, 1),
                dim_x=-2,
            ).transpose(0, 1)

        return z, log_probs, backtrack

    def decode(self, emissions, mask):
        # Forward pass
        backtrack = self.propagate(emissions, mask, ring_op_name="max", use_constraints=True, way="forward")[2]
        path = [backtrack[-1][0, :, 0]]
        # what is happening here ? why did i write:
        # backtrack = torch.stack(backtrack[:-1] + backtrack[-2:-1], 2).squeeze(0)
        # backtrack = torch.stack(backtrack, 2).squeeze(0)
        if len(backtrack) > 1:
            # backtrack = torch.zeros(*backtrack[-1].shape[:-1], self.num_tags, dtype=torch.long)
            backtrack = torch.stack(backtrack[:-1] + backtrack[-2:-1], 2).squeeze(0)
            backtrack[range(len(mask)), mask.sum(1) - 1] = path[-1].unsqueeze(-1)

            # Backward max path following
            for k in range(backtrack.shape[1] - 2, -1, -1):
                path.insert(0, backtrack[:, k][range(len(path[0])), path[0]])
        path = torch.stack(path, -1).masked_fill(~mask, 0)
        return path

    def sample(self, emissions, mask, n):
        # Forward pass
        log_alphas = self.propagate(emissions, mask, ring_op_name="logsumexp", use_constraints=True)[1].transpose(0, 1)

        # Backward sampling
        sequences = []
        bs = len(mask)

        start_transitions = self.start_transitions.masked_fill(self.start_forbidden_transitions, IMPOSSIBLE) if self.start_forbidden_transitions is not None else self.start_transitions
        transitions = self.transitions.masked_fill(self.forbidden_transitions, IMPOSSIBLE) if self.forbidden_transitions is not None else self.transitions
        end_transitions = self.end_transitions.masked_fill(self.end_forbidden_transitions, IMPOSSIBLE) if self.end_forbidden_transitions is not None else self.end_transitions

        # Sample multiple tags for the last token of each sample
        next_log_prob = (
              log_alphas[range(bs), mask.sum(-1) - 1] +
              end_transitions
        )
        next_tag = torch.multinomial(next_log_prob.softmax(-1), n, replacement=True)
        sequences.insert(0, next_tag)

        seq_size = emissions.shape[1]
        for i in range(seq_size - 2, -1, -1):
            next_log_prob = (
                  log_alphas[:, i].unsqueeze(1) +
                  transitions[:, next_tag].permute(1, 2, 0)
            ).softmax(-1)
            next_tag = torch.where(
                mask[:, i + 1].unsqueeze(-1),  # if next token is not a padding token
                torch.multinomial(next_log_prob.reshape(-1, next_log_prob.shape[-1]), 1).reshape(next_tag.shape),  # then put the sampled tags
                next_tag,  # otherwise replicate the tags sampled at the end
            )
            sequences.insert(0, next_tag)
        return torch.stack(sequences, 1).permute(2, 0, 1).masked_fill(~mask.unsqueeze(0), 0)

    def forward(self, emissions, mask, tags, use_constraints=False, reduction="mean"):
        z = self.propagate(emissions, mask, ring_op_name="logsumexp", use_constraints=use_constraints)[0]
        posterior_potential = self.propagate(emissions, mask, tags, ring_op_name="posterior", use_constraints=use_constraints)[0]
        nll = (posterior_potential - z)

        if reduction == 'none':
            return nll
        if reduction == 'sum':
            return nll.sum()
        if reduction == 'mean':
            return nll.mean()
        assert reduction == 'token_mean'
        return nll.sum() / mask.float().sum()

    def marginal(self, emissions, mask):
        z_forward, log_alphas = self.propagate(emissions, mask, ring_op_name="logsumexp", use_constraints=True)[:2]
        log_betas = self.propagate(emissions, mask, ring_op_name="logsumexp", use_constraints=True, way="backward")[1]
        return log_alphas.transpose(0, 1) + log_betas.transpose(0, 1) - emissions - z_forward.squeeze(0).unsqueeze(-1).unsqueeze(-1)


class BIOULDecoder(LinearChainCRF):
    def __init__(self, num_labels, with_start_end_transitions=True, allow_overlap=False, learnable_transitions=True):
        O, I, B, L, U = 0, 1, 2, 3, 4

        num_tags = 1 + num_labels * 4
        forbidden_transitions = torch.ones(num_tags, num_tags, dtype=torch.bool)
        forbidden_transitions[O, O] = 0  # O to O
        for i in range(num_labels):
            STRIDE = 4 * i
            forbidden_transitions[O, B + STRIDE] = 0  # O to B-i
            forbidden_transitions[B + STRIDE, I + STRIDE] = 0  # B-i to I-i
            forbidden_transitions[I + STRIDE, I + STRIDE] = 0  # I-i to I-i
            forbidden_transitions[I + STRIDE, L + STRIDE] = 0  # I-i to L-i
            forbidden_transitions[B + STRIDE, L + STRIDE] = 0  # B-i to L-i
            if allow_overlap:
                forbidden_transitions[L + STRIDE, I + STRIDE] = 0  # L-i to I-i
                forbidden_transitions[L + STRIDE, L + STRIDE] = 0  # L-i to L-i
                forbidden_transitions[L + STRIDE, U + STRIDE] = 0  # L-i to U-i

                forbidden_transitions[I + STRIDE, B + STRIDE] = 0  # I-i to B-i
                forbidden_transitions[B + STRIDE, B + STRIDE] = 0  # B-i to B-i
                forbidden_transitions[U + STRIDE, B + STRIDE] = 0  # U-i to B-i

                forbidden_transitions[B + STRIDE, U + STRIDE] = 0  # B-i to U-i
                forbidden_transitions[U + STRIDE, L + STRIDE] = 0  # U-i to L-i

                forbidden_transitions[U + STRIDE, I + STRIDE] = 0  # U-i to I-i
                forbidden_transitions[I + STRIDE, U + STRIDE] = 0  # I-i to U-i
                forbidden_transitions[U + STRIDE, U + STRIDE] = 0  # I-i to U-i
            forbidden_transitions[L + STRIDE, O] = 0  # L-i to O
            forbidden_transitions[O, U + STRIDE] = 0  # O to U-i
            forbidden_transitions[U + STRIDE, O] = 0  # U-i to O
            for j in range(num_labels):
                STRIDE_J = j * 4
                forbidden_transitions[L + STRIDE, B + STRIDE_J] = 0  # L-i to B-j
                forbidden_transitions[L + STRIDE, U + STRIDE_J] = 0  # L-i to U-j
                forbidden_transitions[U + STRIDE, B + STRIDE_J] = 0  # U-i to B-j

        start_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)
        if with_start_end_transitions:
            for i in range(num_labels):
                STRIDE = 4 * i
                start_forbidden_transitions[I + STRIDE] = 1  # forbidden to start by I-i
                start_forbidden_transitions[L + STRIDE] = 1  # forbidden to start by L-i

        end_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)
        if with_start_end_transitions:
            for i in range(num_labels):
                STRIDE = 4 * i
                end_forbidden_transitions[I + STRIDE] = 1  # forbidden to end by I-i
                end_forbidden_transitions[B + STRIDE] = 1  # forbidden to end by B-i

        super().__init__(forbidden_transitions,
                         start_forbidden_transitions,
                         end_forbidden_transitions,
                         with_start_end_transitions=with_start_end_transitions,
                         learnable_transitions=learnable_transitions)

    @staticmethod
    def spans_to_tags(sample_ids, begins, ends, ner_labels, n_samples, n_tokens):
        BEFORE, AFTER, INSIDE = 0, 1, 2
        positions = torch.arange(n_tokens, device=begins.device).unsqueeze(0)
        begins = begins.unsqueeze(1)
        ends = ends.unsqueeze(1)
        mention_tags = (
            # BEFORE tags
              ((positions < begins) * BEFORE) +
              # AFTER tags
              ((positions >= ends) * AFTER) +
              # INSIDE tags
              (((positions >= begins) & (positions < ends)) * (INSIDE + ner_labels.unsqueeze(1)))
        )
        return torch.zeros((n_samples, n_tokens), dtype=torch.long, device=begins.device).index_add_(0, sample_ids, mention_tags)

    @staticmethod
    def tags_to_spans(tag, mask=None):
        BEFORE, AFTER, INSIDE = 0, 1, 2

        if mask is not None:
            tag = tag.masked_fill(~mask, 0)
        prev_tag = tag.roll(-1)
        prev_tag[..., -1] = AFTER
        label = (tag - 2)
        is_B = (prev_tag == BEFORE) | (tag != BEFORE)
        is_L = (tag == AFTER) | (prev_tag != AFTER)
        begin_tag = is_B.nonzero()
        end_tag = is_L.nonzero()

        span_label = label[is_B]
        spans_count_per_doc = is_B.sum(-1)
        max_spans_count_per_doc = 0 if 0 in begin_tag.shape else spans_count_per_doc.max()

        doc_entity_id = torch.zeros(*spans_count_per_doc.shape, max_spans_count_per_doc, dtype=torch.long, device=tag.device)
        doc_entity_mask = torch.arange(max_spans_count_per_doc, device=tag.device).view(1, -1) < spans_count_per_doc.unsqueeze(-1)
        doc_entity_id[doc_entity_mask] = torch.arange(begin_tag.shape[0], device=tag.device)

        return {
            "doc_spans_id": doc_entity_id,
            "doc_spans_mask": doc_entity_mask,
            "span_begin": begin_tag[:, 1],
            "span_end": end_tag[:, 1],  # for a tag sequence O O B I O, (begin_tag, end_tag) is (2, 3) so we want token span 2:4
            "span_label": span_label,
            "span_doc_id": begin_tag[:, 0],
        }

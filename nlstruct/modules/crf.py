import torch

from nlstruct.utils.torch import torch_global as tg

IMPOSSIBLE = -10000


class LinearChainCRF(torch.nn.Module):
    def __init__(self, num_tags, start_transitions_mask=None, transitions_mask=None, end_transitions_mask=None, with_start_end_transitions=True):
        super().__init__()
        self.num_tags = num_tags

        # Make masks to buffer, since they are not parameters, but we want pytorch to know they are part of
        # the model and should be moved to the new device when module.to(...) is called on the LinearChainCRF
        self.register_buffer('start_transitions_mask', start_transitions_mask)
        self.register_buffer('transitions_mask', transitions_mask)
        self.register_buffer('end_transitions_mask', end_transitions_mask)

        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))

        if with_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.empty(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.empty(num_tags))
        else:
            self.register_buffer('start_transitions', torch.zeros(num_tags))
            self.register_buffer('end_transitions', torch.zeros(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        torch.nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        torch.nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        torch.nn.init.uniform_(self.transitions, -0.1, 0.1)

    def compute_betas(self, emissions, mask):
        """
        Each beta is the potential for the state given all future observations and the current one
        """
        ring_op = lambda x: x.logsumexp(1)

        transitions = self.transitions.t()

        log_probs = [self.end_transitions + emissions[range(len(mask)), mask.sum(1) - 1]]

        for k in range(emissions.shape[1] - 2, -1, -1):
            log_probs.append(torch.where(
                mask[:, k + 1].unsqueeze(-1),
                ring_op(log_probs[-1].unsqueeze(-1) + transitions.unsqueeze(0)) + emissions[:, k],
                log_probs[-1]))

        z = ring_op(log_probs[-1] + self.start_transitions)

        return torch.stack(log_probs[::-1], 1), z

    def decode(self, emissions, mask):
        # Forward pass
        _, _, backtrack = self.recurse_forward(emissions, mask, ring_op_name="max", use_constraints=True)
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
        log_alphas = torch.stack(self.recurse_forward(emissions, mask, ring_op_name="logsumexp", use_constraints=True)[1], 2).squeeze(0)

        # Backward sampling
        sequences = []
        bs = len(mask)

        start_transitions = self.start_transitions.masked_fill(self.start_transitions_mask, IMPOSSIBLE) if self.start_transitions_mask is not None else self.start_transitions
        transitions = self.transitions.masked_fill(self.transitions_mask, IMPOSSIBLE) if self.transitions_mask is not None else self.transitions
        end_transitions = self.end_transitions.masked_fill(self.end_transitions_mask, IMPOSSIBLE) if self.end_transitions_mask is not None else self.end_transitions

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
                torch.multinomial(next_log_prob.reshape(-1, 5), 1).reshape(next_tag.shape),  # then put the sampled tags
                next_tag,  # otherwise replicate the tags sampled at the end
            )
            sequences.insert(0, next_tag)
        return torch.stack(sequences, 1).permute(2, 0, 1).masked_fill(~mask.unsqueeze(0), 0)

    def forward(self, emissions, mask, tags, use_constraints=False, reduction="mean"):
        z = self.recurse_forward(emissions, mask, ring_op_name="logsumexp", use_constraints=use_constraints)[0]
        posterior_potential = self.recurse_forward(emissions, mask, tags, ring_op_name="posterior", use_constraints=use_constraints)[0]
        nll = (posterior_potential - z)

        if reduction == 'none':
            return nll
        if reduction == 'sum':
            return nll.sum()
        if reduction == 'mean':
            return nll.mean()
        assert reduction == 'token_mean'
        return nll.sum() / mask.float().sum()

    def recurse_forward(self, emissions, mask, tags=None, ring_op_name="logsumexp", use_constraints=False):
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
            start_transitions = self.start_transitions.masked_fill(self.start_transitions_mask, IMPOSSIBLE) if self.start_transitions_mask is not None else self.start_transitions
            transitions = self.transitions.masked_fill(self.transitions_mask, IMPOSSIBLE) if self.transitions_mask is not None else self.transitions
            end_transitions = self.end_transitions.masked_fill(self.end_transitions_mask, IMPOSSIBLE) if self.end_transitions_mask is not None else self.end_transitions
        else:
            start_transitions = self.start_transitions
            transitions = self.transitions
            end_transitions = self.end_transitions

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

        return z, log_probs, backtrack


class BIODecoder(LinearChainCRF):
    def __init__(self, num_labels, with_start_end_transitions=True):
        num_tags = 1 + num_labels * 2
        O, B, I = 0, 1, 2
        transitions_mask = torch.ones(num_tags, num_tags, device=tg.device, dtype=torch.bool)
        transitions_mask[O, O] = 0  # O to O
        for i in range(num_labels):
            STRIDE = i * 2
            transitions_mask[O, B + STRIDE] = 0  # O to B-i
            transitions_mask[B + STRIDE, I + STRIDE] = 0  # B-i to I-i
            transitions_mask[I + STRIDE, I + STRIDE] = 0  # I-i to I-i
            transitions_mask[I + STRIDE, O] = 0  # I-i to O
            transitions_mask[B + STRIDE, O] = 0  # B-i to O
            for j in range(num_labels):
                STRIDE_J = j * 2
                transitions_mask[B + STRIDE, B + STRIDE_J] = 0  # B-i to B-j
                transitions_mask[I + STRIDE, B + STRIDE_J] = 0  # I-i to B-j

        start_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        for i in range(num_labels):
            STRIDE = i * 2
            start_transitions_mask[I + STRIDE] = 1  # forbidden to start by I-i

        end_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        super().__init__(num_tags=num_tags,
                         start_transitions_mask=start_transitions_mask,
                         transitions_mask=transitions_mask,
                         end_transitions_mask=end_transitions_mask,
                         with_start_end_transitions=with_start_end_transitions)

    @staticmethod
    def spans_to_tags(sample_ids, begins, ends, ner_labels, n_samples, n_tokens):
        positions = torch.arange(n_tokens, device=begins.device).unsqueeze(0)
        mention_tags = (
            # I tags
              (((positions >= begins.unsqueeze(1)) & (positions < ends.unsqueeze(1))).long() * (ner_labels.unsqueeze(1) * 2 + 2))
              # B tags (= I tag - 1)
              - ((positions == begins.unsqueeze(1)).long() * 1)
        )
        return torch.zeros((n_samples, n_tokens), dtype=torch.long, device=begins.device).index_add_(0, sample_ids, mention_tags)

    @staticmethod
    def tags_to_spans(tag, mask=None):
        if mask is not None:
            tag = tag.masked_fill(~mask, 0)
        unstrided_tags = ((tag - 1) % 2).masked_fill(tag == 0, -1)
        is_B = unstrided_tags == 0
        is_I = unstrided_tags == 1

        label = (tag - 1) // 2
        next_label = label.roll(-1, dims=1)
        next_label[:, -1] = 0
        next_I = is_I.int().roll(-1, dims=1).bool()
        next_I[:, -1] = 0
        begin_tag = is_B.nonzero()
        next_tag = tag.roll(-1, dims=1)
        next_tag[:, -1] = 0
        end_tag = (((tag != next_tag) & is_I) | (is_B & ~((label == next_label) & next_I))).nonzero()

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
            "span_end": end_tag[:, 1] + 1,  # for a tag sequence O O B I O, (begin_tag, end_tag) is (2, 3) so we want token span 2:4
            "span_label": span_label,
            "span_doc_id": begin_tag[:, 0],
        }


class BIOULDecoder(LinearChainCRF):
    def __init__(self, num_labels, with_start_end_transitions=True):
        O, B, I, L, U = 0, 1, 2, 3, 4

        num_tags = 1 + num_labels * 4
        transitions_mask = torch.ones(num_tags, num_tags, device=tg.device, dtype=torch.bool)
        transitions_mask[O, O] = 0  # O to O
        for i in range(num_labels):
            STRIDE = 4 * i
            transitions_mask[O, B + STRIDE] = 0  # O to B-i
            transitions_mask[B + STRIDE, I + STRIDE] = 0  # B-i to I-i
            transitions_mask[I + STRIDE, I + STRIDE] = 0  # I-i to I-i
            transitions_mask[I + STRIDE, L + STRIDE] = 0  # I-i to L-i
            transitions_mask[B + STRIDE, L + STRIDE] = 0  # B-i to L-i
            transitions_mask[L + STRIDE, O] = 0  # L-i to O
            transitions_mask[O, U + STRIDE] = 0  # O to U-i
            transitions_mask[U + STRIDE, O] = 0  # U-i to O
            for j in range(num_labels):
                STRIDE_J = j * 4
                transitions_mask[L + STRIDE, B + STRIDE_J] = 0  # L-i to B-j
                transitions_mask[L + STRIDE, U + STRIDE_J] = 0  # L-i to U-j
                transitions_mask[U + STRIDE, B + STRIDE_J] = 0  # U-i to B-j

        start_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        for i in range(num_labels):
            STRIDE = 4 * i
            start_transitions_mask[I + STRIDE] = 1  # forbidden to start by I-i
            start_transitions_mask[L + STRIDE] = 1  # forbidden to start by L-i

        end_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        for i in range(num_labels):
            STRIDE = 4 * i
            end_transitions_mask[I + STRIDE] = 1  # forbidden to end by I-i
            end_transitions_mask[B + STRIDE] = 1  # forbidden to end by B-i

        super().__init__(num_tags=num_tags,
                         start_transitions_mask=start_transitions_mask,
                         transitions_mask=transitions_mask,
                         end_transitions_mask=end_transitions_mask,
                         with_start_end_transitions=with_start_end_transitions)

    @staticmethod
    def spans_to_tags(sample_ids, begins, ends, ner_labels, n_samples, n_tokens):
        B, I, L, U = 0, 1, 2, 3
        positions = torch.arange(n_tokens, device=begins.device).unsqueeze(0)
        begins = begins.unsqueeze(1)
        ends = ends.unsqueeze(1)
        mention_tags = (
            # I tags
              (((positions >= begins) & (positions < ends)).long() * (1 + ner_labels.unsqueeze(1) * 4 + I))
              # B tags
              + ((positions == begins).long() * (B-I))
              # L tags
              + ((positions == (ends - 1)).long() * (L-I))
              # U tags
              + (((positions == begins) & (positions == (ends - 1))).long() * ((U-I) - (B-I) - (L-I)))
        )
        return torch.zeros((n_samples, n_tokens), dtype=torch.long, device=begins.device).index_add_(0, sample_ids, mention_tags)

    @staticmethod
    def tags_to_spans(tag, mask=None):
        B, I, L, U = 0, 1, 2, 3

        if mask is not None:
            tag = tag.masked_fill(~mask, 0)
        unstrided_tags = ((tag - 1) % 4).masked_fill(tag == 0, -1)
        label = (tag - 1) // 4
        is_B_or_U = (unstrided_tags == B) | (unstrided_tags == U)
        is_L_or_U = (unstrided_tags == L) | (unstrided_tags == U)
        begin_tag = is_B_or_U.nonzero()
        end_tag = is_L_or_U.nonzero()

        span_label = label[is_B_or_U]
        spans_count_per_doc = is_B_or_U.sum(-1)
        max_spans_count_per_doc = 0 if 0 in begin_tag.shape else spans_count_per_doc.max()

        doc_entity_id = torch.zeros(*spans_count_per_doc.shape, max_spans_count_per_doc, dtype=torch.long, device=tag.device)
        doc_entity_mask = torch.arange(max_spans_count_per_doc, device=tag.device).view(1, -1) < spans_count_per_doc.unsqueeze(-1)
        doc_entity_id[doc_entity_mask] = torch.arange(begin_tag.shape[0], device=tag.device)

        return {
            "doc_spans_id": doc_entity_id,
            "doc_spans_mask": doc_entity_mask,
            "span_begin": begin_tag[:, 1],
            "span_end": end_tag[:, 1] + 1,  # for a tag sequence O O B I O, (begin_tag, end_tag) is (2, 3) so we want token span 2:4
            "span_label": span_label,
            "span_doc_id": begin_tag[:, 0],
        }

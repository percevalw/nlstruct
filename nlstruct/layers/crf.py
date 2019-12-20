import torch

from nlstruct.core.torch import torch_global as tg

IMPOSSIBLE = -10000


class CRF(torch.nn.Module):
    def __init__(self, num_tags, start_transitions_mask=None, transitions_mask=None, end_transitions_mask=None):
        super().__init__()
        self.num_tags = num_tags

        self.start_transitions_mask = start_transitions_mask
        self.transitions_mask = transitions_mask
        self.end_transitions_mask = end_transitions_mask

        self.start_transitions = torch.nn.Parameter(torch.empty(num_tags))
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        self.end_transitions = torch.nn.Parameter(torch.empty(num_tags))
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


class BIOULDecoder(CRF):
    def __init__(self):
        num_tags = 5
        TAG_O = 0
        TAG_B = 1
        TAG_I = 2
        TAG_U = 3
        TAG_L = 4
        transitions_mask = torch.zeros(num_tags, num_tags, device=tg.device, dtype=torch.bool)
        transitions_mask[TAG_O, TAG_I] = 1
        transitions_mask[TAG_O, TAG_L] = 1
        transitions_mask[TAG_B, TAG_O] = 1
        transitions_mask[TAG_B, TAG_B] = 1
        transitions_mask[TAG_B, TAG_U] = 1
        transitions_mask[TAG_I, TAG_O] = 1
        transitions_mask[TAG_I, TAG_B] = 1
        transitions_mask[TAG_I, TAG_U] = 1
        transitions_mask[TAG_U, TAG_I] = 1
        transitions_mask[TAG_U, TAG_L] = 1
        transitions_mask[TAG_L, TAG_I] = 1
        transitions_mask[TAG_L, TAG_L] = 1

        start_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        start_transitions_mask[TAG_L] = 1
        start_transitions_mask[TAG_I] = 1

        end_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        end_transitions_mask[TAG_I] = 1
        end_transitions_mask[TAG_B] = 1
        super().__init__(5, start_transitions_mask, transitions_mask, end_transitions_mask)

    @staticmethod
    def extract(tags, tokens):
        squeeze_unflattener = False
        if len(tags.shape) < 3:
            squeeze_unflattener = True
            tags = tags.unsqueeze(0)
        begin_tags = ((tags == 1) | (tags == 3)).nonzero()
        end_tags = ((tags == 4) | (tags == 3)).nonzero()

        if (0 in begin_tags.shape):
            return [None] * 6

        if (0 in begin_tags.shape):
            return [None] * 6

        mentions_count_per_sample = ((tags == 1) | (tags == 3)).sum(-1)
        mentions_unflattener = torch.zeros(*mentions_count_per_sample.shape, max(mentions_count_per_sample.max(), 1), dtype=torch.long)
        mentions_unflattener_mask = torch.arange(max(mentions_count_per_sample.max(), 1), device=tags.device).view(1, 1, -1) < mentions_count_per_sample.unsqueeze(-1)
        mentions_unflattener[mentions_unflattener_mask] = torch.arange(begin_tags.shape[0], device=tags.device)

        mention_length = (end_tags[:, 2] - begin_tags[:, 2]).max().item() + 1
        mentions_from_sequences_col_indexer = torch.arange(mention_length, device=tags.device).unsqueeze(0) + begin_tags[:, 2].unsqueeze(1)
        mentions_from_sequences_row_indexer = begin_tags[:, 1].unsqueeze(1)
        mentions_tokens = tokens[mentions_from_sequences_row_indexer, torch.min(mentions_from_sequences_col_indexer, torch.tensor(tokens.shape[1], device=tags.device) - 1)]
        mentions_tokens_mask = mentions_from_sequences_col_indexer <= end_tags[:, 2].unsqueeze(1)
        if squeeze_unflattener:
            mentions_unflattener = mentions_unflattener.squeeze(0)
            mentions_unflattener_mask = mentions_unflattener_mask.squeeze(0)
        return mentions_tokens, mentions_tokens_mask, mentions_unflattener, mentions_unflattener_mask, begin_tags[:, 2], end_tags[:, 2]


class BIODecoder(CRF):
    def __init__(self, num_labels):
        num_tags = 1 + num_labels * 2
        TAG_O = 0
        TAG_B = 1
        TAG_I = 2
        transitions_mask = torch.ones(num_tags, num_tags, device=tg.device, dtype=torch.bool)
        transitions_mask[0, 0] = 0  # O to O
        for i in range(num_labels):
            transitions_mask[0, 1 + 2 * i] = 0  # O to B-i
            transitions_mask[1 + 2 * i, 2 + 2 * i] = 0  # B-i to I-i
            transitions_mask[2 + 2 * i, 2 + 2 * i] = 0  # I-i to I-i
            transitions_mask[2 + 2 * i, 0] = 0  # I-i to O
            transitions_mask[1 + 2 * i, 0] = 0  # B-i to O
            for j in range(num_labels):
                transitions_mask[1 + 2 * i, 1 + 2 * j] = 0  # B-i to B-j
                transitions_mask[2 + 2 * i, 1 + 2 * j] = 0  # I-i to B-j

        start_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        for i in range(num_labels):
            start_transitions_mask[2 + 2 * i] = 1  # forbidden to start by I-i

        end_transitions_mask = torch.zeros(num_tags, device=tg.device, dtype=torch.bool)
        super().__init__(num_tags=num_tags,
                         start_transitions_mask=start_transitions_mask,
                         transitions_mask=transitions_mask,
                         end_transitions_mask=end_transitions_mask)

    @staticmethod
    def extract(tag, tokens=None):
        squeeze_unflattener = False
        if len(tag.shape) < 3:
            squeeze_unflattener = True
            tag = tag.unsqueeze(0)
        is_B = ((tag - 1) % 2) == 0
        is_I = ((tag - 1) % 2 == 1) & (tag != 0)
        label = (tag - 1) // 2
        next_label = label.roll(-1, dims=2)
        next_label[:, :, -1] = 0
        next_I = is_I.roll(-1, dims=2)
        next_I[:, :, -1] = 0
        begin_tag = is_B.nonzero()
        next_tag = tag.roll(-1, dims=2)
        next_tag[:, :, -1] = 0
        end_tag = (((tag != next_tag) & is_I) | (is_B & ~((label == next_label) & next_I))).nonzero()

        mention_label = label[is_B]

        mentions_count_per_sample = is_B.sum(-1)

        if (0 in begin_tag.shape):
            max_mentions_count_per_sample = 0
            mention_length = 1
        else:
            max_mentions_count_per_sample = mentions_count_per_sample.max()
            mention_length = (end_tag[:, 2] - begin_tag[:, 2]).max().item() + 1

        sample_entity_id = torch.zeros(*mentions_count_per_sample.shape, max(max_mentions_count_per_sample, 1), dtype=torch.long)
        sample_entity_mask = torch.arange(max(max_mentions_count_per_sample, 1), device=tag.device).view(1, 1, -1) < mentions_count_per_sample.unsqueeze(-1)
        sample_entity_id[sample_entity_mask] = torch.arange(begin_tag.shape[0], device=tag.device)

        mentions_from_sequences_col_indexer = torch.arange(mention_length, device=tag.device).unsqueeze(0) + begin_tag[:, 2].unsqueeze(1)
        mentions_from_sequences_row_indexer = begin_tag[:, 1].unsqueeze(1)
        if tokens is not None:
            mentions_tokens = tokens[mentions_from_sequences_row_indexer, torch.min(mentions_from_sequences_col_indexer, torch.tensor(tokens.shape[1], device=tag.device) - 1)]
            mentions_tokens_mask = mentions_from_sequences_col_indexer <= end_tag[:, 2].unsqueeze(1)
        else:
            mentions_tokens = None
            mentions_tokens_mask = None
        if squeeze_unflattener:
            sample_entity_id = sample_entity_id.squeeze(0)
            sample_entity_mask = sample_entity_mask.squeeze(0)

        return {
            "begins": begin_tag[:, 2],
            "ends": end_tag[:, 2] + 1,
            "labels": mention_label,
            "mentions_tokens": mentions_tokens,
            "mentions_tokens_mask": mentions_tokens_mask,
            "sample_entity_id": sample_entity_id,
            "sample_entity_mask": sample_entity_mask,
            "entity_sample_id": begin_tag[:, 1],
        }

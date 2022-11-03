import torch
from utils.constants import START_TOKEN, SPECIAL_TOKENS_INDEX, END_TOKEN


class Beam:
    def __init__(self, beam_size=8, min_length=0, n_top=1, ranker=None,
                 start_token_id=SPECIAL_TOKENS_INDEX[START_TOKEN],
                 end_token_id=SPECIAL_TOKENS_INDEX[END_TOKEN]):
        # Q: What does beam size mean?
        # A: This indicates the size of potential paths to traverse when looking for
        # the best tokens to generate.
        # https://stackoverflow.com/questions/22273119/what-does-the-beam-size-represent-in-the-beam-search-algorithm
        # Q: How is beam_size different from n_top?
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker

        self.end_token_id = end_token_id
        self.top_sentence_ended = False

        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)]

        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []

        self.all_attentions = []
        self.finished = []

        self.n_top = n_top
        self.ranker = ranker

    def get_current_state(self):
        return self.next_ys[-1]

    def advance(self, next_log_probs, current_attention):
        # Q: How does beam search work?
        # A: https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24
        # https://huggingface.co/blog/how-to-generate
        vocabulary_size = next_log_probs.size(1)
        current_length = len(self.next_ys)

        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                # Q: Why are we assigning this value to the end_token_id?
                # A: This is so that it has very low probability weight when calculating the beam
                # search.
                next_log_probs[beam_index][self.end_token_id] = -1e10

        if len(self.prev_ks) > 0:
            # Q: Should this be an addition? I thought beam scores were supposed to be multiplications
            # of probability for each step taken into the search algo
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -1e10
        else:
            beam_scores = next_log_probs[0]

        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(k=self.beam_size, dim=0, largest=True, sorted=True)

        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)

        # Q: Why are they dividing by vocabulary size here? Are they trying to scale from 0 to 1?
        # Now that I think of it, probably trying to keep memory usage low?
        # Definitely not to keep memory usage low.
        # prev_k = top_score_ids / vocabulary_size
        prev_k = top_score_ids
        # Q: Then they reverse the prev_k here. I don't see why next_y should not be zero.
        # I: Indeed it becomes zero, except that since the division for prev_k leads to float precision issues,
        # some values in the next_y tensor are no longer proper zeros
        # next_y = top_score_ids - prev_k * vocabulary_size
        next_y = top_score_ids

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)

        # Q: Interesting. Why are they storing attentions?
        # I: I don't think it is being used at all.
        # prev_attention = current_attention.index_select(dim=0, index=top_score_ids)
        # self.all_attentions.append(prev_attention)

        for beam_index, last_token_id in enumerate(next_y):
            # I: I think this is really cool, stopping the beam search when the max score for that beam path
            # is an end token.
            if last_token_id == self.end_token_id:
                self.finished.append((self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))

        # Q: What effect top_sentence_ended have?
        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_origin(self):
        return self.prev_ks[-1]

    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            while len(self.finished) < minimum:
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1), i)
                i += 1

        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hypothesis(self, timestamp, k):
        hypothesis, attentions = [], []
        for j in range(len(self.prev_ks[:timestamp]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            attentions.append(self.all_attentions[j][k, :, :])
            k = self.prev_ks[j][k]
        attentions_tensor = torch.stack(attentions[::-1]).squeeze(1)
        return hypothesis[::-1], attentions_tensor





        








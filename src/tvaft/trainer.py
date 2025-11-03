# src/tvaft/trainer.py

from torch.nn import CrossEntropyLoss
from transformers import Trainer


class TVAFTTrainer(Trainer):
    """
    Custom trainer for the Token Value-Aware Fine-Tuning (TVAFT) method.
    Overrides the `compute_loss` method to apply saliency weights to each token.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.is_in_train:
            labels = inputs.pop("labels")
            saliency_weights = inputs.pop("saliency_weights")
            inputs.pop("is_correct", None)

            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_saliency_weights = saliency_weights[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            token_wise_ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_wise_ce_loss = token_wise_ce_loss.view_as(shift_labels)

            weighted_loss_per_token = shift_saliency_weights * token_wise_ce_loss

            valid_labels_mask = (shift_labels != -100)
            valid_weighted_losses = weighted_loss_per_token[valid_labels_mask]
            valid_weights = shift_saliency_weights[valid_labels_mask]
            final_loss = valid_weighted_losses.sum() / valid_weights.sum().clamp(min=1e-8)

            return (final_loss, outputs) if return_outputs else final_loss

        else:
            inputs.pop("saliency_weights", None)
            inputs.pop("is_correct", None)
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

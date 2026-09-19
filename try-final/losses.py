import torch
import torch.nn as nn


class ColorizationLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, predicted_ab, real_ab):

        error = torch.abs(
            predicted_ab - real_ab
        )


        color_strength = torch.sqrt(
            real_ab[:, 0:1] ** 2
            +
            real_ab[:, 1:2] ** 2
            +
            1e-8
        )


        weights = 1.0 + 4.0 * color_strength


        weighted_loss = (
            error * weights
        ).mean()


        predicted_strength = torch.sqrt(
            predicted_ab[:, 0] ** 2
            +
            predicted_ab[:, 1] ** 2
            +
            1e-8
        )

        real_strength = torch.sqrt(
            real_ab[:, 0] ** 2
            +
            real_ab[:, 1] ** 2
            +
            1e-8
        )


        strength_loss = torch.abs(
            predicted_strength - real_strength
        ).mean()

        # FINAL LOSS

        total_loss = (
            weighted_loss
            +
            0.15 * strength_loss
        )


        return total_loss
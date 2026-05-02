from __future__ import annotations

import random

import cv2
import numpy as np


def random_resize(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    backward_game_motion: np.ndarray,
    forward_game_motion: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        backward_game_motion = cv2.resize(
            backward_game_motion,
            dsize=None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_LINEAR,
        ) * 2.0
        forward_game_motion = cv2.resize(
            forward_game_motion,
            dsize=None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_LINEAR,
        ) * 2.0

    return img0, imgt, img1, backward_game_motion, forward_game_motion


def random_crop(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    backward_game_motion: np.ndarray,
    forward_game_motion: np.ndarray,
    crop_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crop_height, crop_width = crop_size
    image_height, image_width, _ = img0.shape
    offset_x = np.random.randint(0, image_height - crop_height + 1)
    offset_y = np.random.randint(0, image_width - crop_width + 1)

    img0 = img0[offset_x : offset_x + crop_height, offset_y : offset_y + crop_width, :]
    imgt = imgt[offset_x : offset_x + crop_height, offset_y : offset_y + crop_width, :]
    img1 = img1[offset_x : offset_x + crop_height, offset_y : offset_y + crop_width, :]
    backward_game_motion = backward_game_motion[
        offset_x : offset_x + crop_height,
        offset_y : offset_y + crop_width,
        :,
    ]
    forward_game_motion = forward_game_motion[
        offset_x : offset_x + crop_height,
        offset_y : offset_y + crop_width,
        :,
    ]
    return img0, imgt, img1, backward_game_motion, forward_game_motion


def random_reverse_channel(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    backward_game_motion: np.ndarray,
    forward_game_motion: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]

    return img0, imgt, img1, backward_game_motion, forward_game_motion


def random_vertical_flip(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    backward_game_motion: np.ndarray,
    forward_game_motion: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
        backward_game_motion = backward_game_motion[::-1]
        forward_game_motion = forward_game_motion[::-1]
        backward_game_motion = np.concatenate(
            (
                backward_game_motion[:, :, 0:1],
                -backward_game_motion[:, :, 1:2],
                backward_game_motion[:, :, 2:3],
                -backward_game_motion[:, :, 3:4],
            ),
            axis=2,
        )
        forward_game_motion = np.concatenate(
            (
                forward_game_motion[:, :, 0:1],
                -forward_game_motion[:, :, 1:2],
                forward_game_motion[:, :, 2:3],
                -forward_game_motion[:, :, 3:4],
            ),
            axis=2,
        )

    return img0, imgt, img1, backward_game_motion, forward_game_motion


def random_horizontal_flip(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    backward_game_motion: np.ndarray,
    forward_game_motion: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
        backward_game_motion = backward_game_motion[:, ::-1]
        forward_game_motion = forward_game_motion[:, ::-1]
        backward_game_motion = np.concatenate(
            (
                -backward_game_motion[:, :, 0:1],
                backward_game_motion[:, :, 1:2],
                -backward_game_motion[:, :, 2:3],
                backward_game_motion[:, :, 3:4],
            ),
            axis=2,
        )
        forward_game_motion = np.concatenate(
            (
                -forward_game_motion[:, :, 0:1],
                forward_game_motion[:, :, 1:2],
                -forward_game_motion[:, :, 2:3],
                forward_game_motion[:, :, 3:4],
            ),
            axis=2,
        )

    return img0, imgt, img1, backward_game_motion, forward_game_motion


def random_rotate(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    backward_game_motion: np.ndarray,
    forward_game_motion: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
        backward_game_motion = backward_game_motion.transpose((1, 0, 2))
        forward_game_motion = forward_game_motion.transpose((1, 0, 2))
        backward_game_motion = np.concatenate(
            (
                backward_game_motion[:, :, 1:2],
                backward_game_motion[:, :, 0:1],
                backward_game_motion[:, :, 3:4],
                backward_game_motion[:, :, 2:3],
            ),
            axis=2,
        )
        forward_game_motion = np.concatenate(
            (
                forward_game_motion[:, :, 1:2],
                forward_game_motion[:, :, 0:1],
                forward_game_motion[:, :, 3:4],
                forward_game_motion[:, :, 2:3],
            ),
            axis=2,
        )

    return img0, imgt, img1, backward_game_motion, forward_game_motion


def random_reverse_time(
    img0: np.ndarray,
    imgt: np.ndarray,
    img1: np.ndarray,
    backward_game_motion: np.ndarray,
    forward_game_motion: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.uniform(0, 1) < p:
        img0, img1 = img1, img0
        backward_game_motion = np.concatenate(
            (backward_game_motion[:, :, 2:4], backward_game_motion[:, :, 0:2]),
            axis=2,
        )
        forward_game_motion = np.concatenate(
            (forward_game_motion[:, :, 2:4], forward_game_motion[:, :, 0:2]),
            axis=2,
        )

    return img0, imgt, img1, backward_game_motion, forward_game_motion

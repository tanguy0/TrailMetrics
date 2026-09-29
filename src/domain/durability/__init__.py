"""Durability: how the energetic cost of running drifts over a long effort.

``phi(t) >= 1`` multiplies the fresh cost of every route point (see
:mod:`src.domain.durability.model`). One hierarchical model — a population prior
plus an individually shrunk offset (:mod:`.personalization`) — fed back into the
race planner by :mod:`.solver`, and fitted offline by :mod:`.calibration`.

"Durability" rather than "fatigue" on purpose: in this app *fatigue* is the
Banister acute-load series (:mod:`src.domain.dataset.training_load`).
"""

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure, reusable Maximal Marginal Relevance (MMR) greedy selection — the single implementation shared by the
 * pipeline MMR response processor ({@link MMRRerankProcessor}) and the {@code diversify} retriever.
 * <p>
 * It is deliberately decoupled from {@code SearchHit} and {@code _source}: callers pass an explicit candidate
 * order (ids), an {@code id -> vector} map (vectors obtained however the caller likes — {@code _source} parse
 * for the pipeline, {@code docvalue_fields} for the retriever), and an {@code id -> score} map. The helper
 * returns the selected ids in MMR selection order.
 * <p>
 * <b>Formula.</b> At each step it picks the candidate maximizing
 * {@code (1 - diversity) * score(c) - diversity * maxSim(c, selected)} where {@code maxSim} is the greatest
 * similarity between {@code c} and any already-selected doc ({@code 0} when nothing is selected yet).
 * Pairwise similarities are memoized in a symmetric cache so each unordered pair is compared once.
 * <p>
 * <b>Determinism.</b> Candidates are considered in the caller-supplied {@code orderedIds} order and ties are
 * broken by keeping the earlier candidate in that order (strict {@code >} comparison). Callers that need a
 * reproducible result across shards/nodes (e.g. the retriever on a multi-shard cluster) must pass
 * {@code orderedIds} in a deterministic order (by child {@code position}, then {@code (index, shardId, id)}).
 * <p>
 * <b>Missing vectors.</b> A candidate whose id has no entry in {@code idToVector} (or a {@code null} vector)
 * is treated as incomparable: its {@code maxSim} contribution is {@code 0} for every pairing, so it is
 * <i>kept</i> and competes on pure relevance {@code (1 - diversity) * score}. It is never dropped and never
 * errors. This is the only correct behavior for the one case it arises — a non-vector leg (e.g. BM25) under
 * a {@code diversify(rank_fusion(...))}: such a doc is genuinely not a near-duplicate of anything.
 */
public final class MMRSelector {

    private MMRSelector() {}

    /**
     * Run greedy MMR selection.
     *
     * @param orderedIds         candidate ids in deterministic consideration order (not mutated)
     * @param idToVector         id -&gt; vector ({@code float[]} or {@code byte[]}); a missing/null entry means
     *                           "no vector" (treated as incomparable, {@code maxSim = 0})
     * @param idToScore          id -&gt; relevance score
     * @param similarityFunction similarity used for {@code maxSim} (higher = more similar for every space)
     * @param diversity          the diversity weight in {@code [0,1]} ({@code 1 - lambda}); {@code 0} = pure
     *                           relevance (order unchanged), {@code 1} = pure diversity
     * @param outputSize         number of ids to select (clamped to the number of candidates)
     * @param isFloatVector      {@code true} for {@code float[]} vectors, {@code false} for {@code byte[]}
     * @param explainOut         optional, insertion-ordered; when non-null, receives an {@link MMRExplainInfo}
     *                           per selected id at the moment it was chosen
     * @return the selected ids, in MMR selection order
     */
    public static List<String> select(
        final List<String> orderedIds,
        final Map<String, Object> idToVector,
        final Map<String, Float> idToScore,
        final KNNVectorSimilarityFunction similarityFunction,
        final float diversity,
        final int outputSize,
        final boolean isFloatVector,
        final Map<String, MMRExplainInfo> explainOut
    ) {
        final List<String> candidates = new ArrayList<>(orderedIds);
        final List<String> selected = new ArrayList<>(Math.min(outputSize, candidates.size()));
        final Map<String, Float> simCache = new HashMap<>();
        final boolean collectExplain = explainOut != null;

        final int target = Math.min(outputSize, candidates.size());
        while (selected.size() < target && candidates.isEmpty() == false) {
            String bestId = null;
            int bestIndex = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            float bestMaxSim = 0.0f;

            for (int i = 0; i < candidates.size(); i++) {
                final String candidateId = candidates.get(i);
                final float maxSim = maxSimilarityToSelected(
                    candidateId,
                    selected,
                    idToVector,
                    similarityFunction,
                    isFloatVector,
                    simCache
                );
                final float score = idToScore.getOrDefault(candidateId, 0.0f);
                final double mmr = (1 - diversity) * score - diversity * maxSim;
                // Strict > keeps the earlier candidate (deterministic tie-break on orderedIds order).
                if (mmr > bestScore) {
                    bestScore = mmr;
                    bestMaxSim = maxSim;
                    bestId = candidateId;
                    bestIndex = i;
                }
            }

            if (bestId == null) {
                break;
            }
            if (collectExplain) {
                explainOut.put(
                    bestId,
                    MMRExplainInfo.builder()
                        .originalScore(idToScore.getOrDefault(bestId, 0.0f))
                        .maxSimilarityToSelected(bestMaxSim)
                        .mmrScore(bestScore)
                        .diversity(diversity)
                        .build()
                );
            }
            selected.add(bestId);
            candidates.remove(bestIndex);
        }

        return selected;
    }

    private static float maxSimilarityToSelected(
        final String candidateId,
        final List<String> selected,
        final Map<String, Object> idToVector,
        final KNNVectorSimilarityFunction similarityFunction,
        final boolean isFloatVector,
        final Map<String, Float> simCache
    ) {
        final Object candidateVector = idToVector.get(candidateId);
        // No vector for the candidate → incomparable → contributes no similarity penalty.
        if (candidateVector == null) {
            return 0.0f;
        }
        float maxSim = 0.0f;
        for (final String selId : selected) {
            final Object selectedVector = idToVector.get(selId);
            if (selectedVector == null) {
                continue; // selected doc has no vector → not comparable to this candidate
            }
            final String key = candidateId + ":" + selId;
            final Float cached = simCache.get(key);
            final float sim;
            if (cached != null) {
                sim = cached;
            } else {
                if (isFloatVector) {
                    sim = similarityFunction.compare((float[]) candidateVector, (float[]) selectedVector);
                } else {
                    sim = similarityFunction.compare((byte[]) candidateVector, (byte[]) selectedVector);
                }
                simCache.put(key, sim);
                simCache.put(selId + ":" + candidateId, sim); // symmetric
            }
            maxSim = Math.max(maxSim, sim);
        }
        return maxSim;
    }
}

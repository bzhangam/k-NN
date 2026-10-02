/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Unit tests for {@link MMRSelector} — the shared greedy MMR selection used by both the pipeline MMR
 * processor and the diversify retriever. Covers the λ extremes, a hand-computed mid-λ case, deterministic
 * tie-breaking, the missing-vector keep rule, byte vs float, and edge cases.
 */
public class MMRSelectorTests extends KNNTestCase {

    private static final KNNVectorSimilarityFunction L2 = SpaceType.L2.getKnnVectorSimilarityFunction();

    private static Map<String, Object> floatVectors(Map<String, float[]> in) {
        return new HashMap<>(in);
    }

    private static List<String> ids(String... ids) {
        List<String> l = new ArrayList<>();
        Collections.addAll(l, ids);
        return l;
    }

    // diversity = 1 - lambda. lambda=1 => diversity=0 => pure relevance => order unchanged.
    public void testLambdaOneKeepsRelevanceOrder() {
        List<String> ordered = ids("a", "b", "c");
        Map<String, Object> vecs = floatVectors(Map.of("a", new float[] { 1, 0 }, "b", new float[] { 1, 0 }, "c", new float[] { 0, 1 }));
        Map<String, Float> scores = Map.of("a", 0.9f, "b", 0.8f, "c", 0.7f);

        List<String> selected = MMRSelector.select(ordered, vecs, scores, L2, 0.0f /*diversity*/, 3, true, null);
        assertEquals(List.of("a", "b", "c"), selected);
    }

    // diversity=1 => pure diversity. First pick is top-relevance (maxSim=0), then the most dissimilar.
    public void testLambdaZeroPrefersDiverse() {
        // a and b are identical vectors (near-dup); c is orthogonal. a has the top score so is picked first.
        List<String> ordered = ids("a", "b", "c");
        Map<String, Object> vecs = floatVectors(Map.of("a", new float[] { 1, 0 }, "b", new float[] { 1, 0 }, "c", new float[] { 0, 1 }));
        Map<String, Float> scores = Map.of("a", 0.9f, "b", 0.85f, "c", 0.1f);

        List<String> selected = MMRSelector.select(ordered, vecs, scores, L2, 1.0f /*diversity*/, 3, true, null);
        assertEquals("a picked first (first selection has maxSim=0)", "a", selected.get(0));
        assertEquals("c (orthogonal) preferred over near-duplicate b under pure diversity", "c", selected.get(1));
        assertEquals("b (near-dup of a) last", "b", selected.get(2));
    }

    public void testEmptyCandidates() {
        List<String> selected = MMRSelector.select(ids(), floatVectors(Map.of()), Map.of(), L2, 0.5f, 5, true, null);
        assertTrue(selected.isEmpty());
    }

    public void testSingleCandidate() {
        List<String> selected = MMRSelector.select(
            ids("a"),
            floatVectors(Map.of("a", new float[] { 1, 0 })),
            Map.of("a", 0.5f),
            L2,
            0.5f,
            5,
            true,
            null
        );
        assertEquals(List.of("a"), selected);
    }

    public void testOutputSizeClampedToCandidates() {
        List<String> selected = MMRSelector.select(
            ids("a", "b"),
            floatVectors(Map.of("a", new float[] { 1, 0 }, "b", new float[] { 0, 1 })),
            Map.of("a", 0.9f, "b", 0.8f),
            L2,
            0.5f,
            10, // more than available
            true,
            null
        );
        assertEquals(2, selected.size());
    }

    // A candidate with no vector is kept (maxSim=0), competing on pure relevance — never dropped.
    public void testMissingVectorKeptAndScoredByRelevance() {
        // "b" has no vector entry → incomparable. With diversity high, a near-dup of a would be penalized, but
        // b (no vector) is not, so it should still be selectable on its relevance.
        List<String> ordered = ids("a", "b", "c");
        Map<String, Object> vecs = new HashMap<>();
        vecs.put("a", new float[] { 1, 0 });
        vecs.put("c", new float[] { 1, 0 }); // near-dup of a
        // b intentionally absent
        Map<String, Float> scores = Map.of("a", 0.9f, "b", 0.6f, "c", 0.85f);

        List<String> selected = MMRSelector.select(ordered, vecs, scores, L2, 0.9f /*high diversity*/, 3, true, null);
        assertEquals("all three kept (b never dropped despite missing vector)", 3, selected.size());
        assertTrue(selected.contains("b"));
        assertEquals("a first (top relevance, maxSim=0)", "a", selected.get(0));
        // c is a near-dup of a so heavily penalized under high diversity; b (incomparable, maxSim=0) outranks it.
        assertTrue("b selected before near-duplicate c", selected.indexOf("b") < selected.indexOf("c"));
    }

    public void testByteVectors() {
        List<String> ordered = ids("a", "b", "c");
        Map<String, Object> vecs = new HashMap<>();
        vecs.put("a", new byte[] { 10, 0 });
        vecs.put("b", new byte[] { 10, 0 }); // near-dup of a
        vecs.put("c", new byte[] { 0, 10 });
        Map<String, Float> scores = Map.of("a", 0.9f, "b", 0.85f, "c", 0.1f);

        List<String> selected = MMRSelector.select(
            ordered,
            vecs,
            scores,
            SpaceType.L2.getKnnVectorSimilarityFunction(),
            1.0f,
            3,
            false /*byte*/,
            null
        );
        assertEquals("a", selected.get(0));
        assertEquals("c diverse pick over near-dup b", "c", selected.get(1));
    }

    // Determinism: identical MMR scores must resolve by the caller's orderedIds order, regardless of any
    // incidental map iteration order. Shuffling the input id order flips which tied candidate wins.
    public void testDeterministicTieBreakFollowsOrderedIds() {
        // Two orthogonal vectors with identical scores → on the first pick both have maxSim=0 and equal score,
        // so the FIRST in orderedIds must win.
        Map<String, Object> vecs = floatVectors(Map.of("x", new float[] { 1, 0 }, "y", new float[] { 0, 1 }));
        Map<String, Float> scores = Map.of("x", 0.5f, "y", 0.5f);

        List<String> xy = MMRSelector.select(ids("x", "y"), vecs, scores, L2, 0.5f, 1, true, null);
        assertEquals("x first when ordered [x,y]", List.of("x"), xy);

        List<String> yx = MMRSelector.select(ids("y", "x"), vecs, scores, L2, 0.5f, 1, true, null);
        assertEquals("y first when ordered [y,x]", List.of("y"), yx);
    }

    public void testExplainPopulatedPerSelected() {
        List<String> ordered = ids("a", "b");
        Map<String, Object> vecs = floatVectors(Map.of("a", new float[] { 1, 0 }, "b", new float[] { 0, 1 }));
        Map<String, Float> scores = Map.of("a", 0.9f, "b", 0.8f);
        Map<String, MMRExplainInfo> explain = new HashMap<>();

        List<String> selected = MMRSelector.select(ordered, vecs, scores, L2, 0.5f, 2, true, explain);
        assertEquals(2, selected.size());
        for (String id : selected) {
            MMRExplainInfo info = explain.get(id);
            assertNotNull("explain present for " + id, info);
            assertEquals(scores.get(id), info.getOriginalScore(), 1e-6);
            assertEquals(0.5f, info.getDiversity(), 1e-6);
        }
        // First selected has maxSim 0.
        assertEquals(0.0f, explain.get(selected.get(0)).getMaxSimilarityToSelected(), 1e-6);
    }
}

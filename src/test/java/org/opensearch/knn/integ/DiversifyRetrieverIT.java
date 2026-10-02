/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.List;
import java.util.Map;

/**
 * Integration tests for the {@code diversify} (MMR) retriever against a real cluster, driving the
 * {@code _search} retriever path end-to-end: SPI registration, the {@code docvalue_fields} vector
 * ride-along, and MMR selection over native Faiss / Lucene vectors.
 * <p>
 * Corpus: a 2-D {@code knn_vector} field {@code embedding}. Docs a and b are near-duplicates (same
 * direction, a slightly more relevant); c is orthogonal (diverse). A plain knn query ranks the near-dups
 * a,b at the top; diversify should surface the diverse doc c earlier.
 */
public class DiversifyRetrieverIT extends KNNRestTestCase {

    private static final String FIELD = "embedding";
    private static final int DIM = 2;

    @SneakyThrows
    private void createIndex(String index, KNNEngine engine) {
        // 3 shards / 0 replicas (multi-shard coordinator-side selection), knn enabled.
        Settings settings = Settings.builder().put("index.knn", true).put("number_of_shards", 3).put("number_of_replicas", 0).build();
        // Mapping body is the mapping definition only (properties); settings are passed separately.
        String mapping = "{\"properties\":{\""
            + FIELD
            + "\":{\"type\":\"knn_vector\",\"dimension\":"
            + DIM
            + ",\"method\":{\"name\":\"hnsw\",\"space_type\":\""
            + SpaceType.L2.getValue()
            + "\",\"engine\":\""
            + engine.getName()
            + "\"}},\"title\":{\"type\":\"text\"}}}";
        createKnnIndex(index, settings, mapping);
    }

    @SneakyThrows
    private void indexCorpus(String index) {
        // a,b near-duplicates (direction ~[1,0]); c orthogonal (~[0,1]).
        addKnnDoc(index, "a", FIELD, new Float[] { 1.00f, 0.00f });
        addKnnDoc(index, "b", FIELD, new Float[] { 0.98f, 0.02f });
        addKnnDoc(index, "c", FIELD, new Float[] { 0.00f, 1.00f });
        refreshIndex(index);
    }

    // Corpus variant where every doc also carries a "title" so a BM25 (match) leg can score them — used by
    // the per-leg under-fusion test. a,b near-duplicate vectors; c orthogonal. All share a title term so the
    // bm25 leg returns the same docs the knn leg does.
    @SneakyThrows
    private void indexCorpusWithTitles(String index) {
        addKnnDoc(index, "a", "{\"" + FIELD + "\":[1.00,0.00],\"title\":\"widget alpha\"}");
        addKnnDoc(index, "b", "{\"" + FIELD + "\":[0.98,0.02],\"title\":\"widget beta\"}");
        addKnnDoc(index, "c", "{\"" + FIELD + "\":[0.00,1.00],\"title\":\"widget gamma\"}");
        refreshIndex(index);
    }

    private String diversifyBody(float lambda, boolean sourceFalse, boolean explain) {
        // Query vector near a/b so a,b are the top organic knn matches; c is farther.
        String knn = "{\"standard\":{\"query\":{\"knn\":{\"" + FIELD + "\":{\"vector\":[1.0,0.0],\"k\":10}}}}}";
        StringBuilder sb = new StringBuilder();
        sb.append("{\"retriever\":{\"diversify\":{")
            .append("\"vector_field\":\"")
            .append(FIELD)
            .append("\",")
            .append("\"lambda\":")
            .append(lambda)
            .append(",")
            .append("\"window_size\":10,")
            .append("\"retriever\":")
            .append(knn)
            .append("}},\"size\":3");
        if (sourceFalse) {
            sb.append(",\"_source\":false");
        }
        if (explain) {
            sb.append(",\"explain\":true");
        }
        sb.append("}");
        return sb.toString();
    }

    @SneakyThrows
    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> search(String index, String body) {
        Response response = searchKNNIndex(index, body, 10);
        Map<String, Object> map = entityAsMap(response);
        Map<String, Object> hitsWrapper = (Map<String, Object>) map.get("hits");
        return (List<Map<String, Object>>) hitsWrapper.get("hits");
    }

    private static List<String> ids(List<Map<String, Object>> hits) {
        return hits.stream().map(h -> (String) h.get("_id")).toList();
    }

    // DIV-IT-1 / DIV-IT-5: lambda low (high diversity) surfaces the diverse doc c ahead of the near-dup b.
    @SneakyThrows
    public void testDiversifySeparatesNearDuplicates_faiss() {
        String index = "diversify_faiss";
        createIndex(index, KNNEngine.FAISS);
        indexCorpus(index);

        List<String> ids = ids(search(index, diversifyBody(0.1f, false, false)));
        assertEquals("all three returned", 3, ids.size());
        assertEquals("top organic match a first", "a", ids.get(0));
        assertTrue("diverse doc c promoted ahead of near-duplicate b", ids.indexOf("c") < ids.indexOf("b"));
        deleteKNNIndex(index);
    }

    // DIV-IT-8: same behavior on the Lucene engine.
    @SneakyThrows
    public void testDiversifySeparatesNearDuplicates_lucene() {
        String index = "diversify_lucene";
        createIndex(index, KNNEngine.LUCENE);
        indexCorpus(index);

        List<String> ids = ids(search(index, diversifyBody(0.1f, false, false)));
        assertEquals(3, ids.size());
        assertEquals("a", ids.get(0));
        assertTrue("diverse c ahead of near-dup b", ids.indexOf("c") < ids.indexOf("b"));
        deleteKNNIndex(index);
    }

    // DIV-IT-4: lambda=1 (pure relevance) keeps the plain knn order (near-dups a,b above diverse c).
    @SneakyThrows
    public void testLambdaOneKeepsRelevanceOrder() {
        String index = "diversify_lambda1";
        createIndex(index, KNNEngine.FAISS);
        indexCorpus(index);

        List<String> ids = ids(search(index, diversifyBody(1.0f, false, false)));
        assertEquals(List.of("a", "b", "c"), ids);
        deleteKNNIndex(index);
    }

    // DIV-IT-6: _source:false — vector comes via docvalue_fields ride-along; diversify still works and the
    // response carries no _source.
    @SneakyThrows
    public void testRideAlongWithSourceDisabled() {
        String index = "diversify_nosource";
        createIndex(index, KNNEngine.FAISS);
        indexCorpus(index);

        List<Map<String, Object>> hits = search(index, diversifyBody(0.1f, true, false));
        assertEquals(3, hits.size());
        for (Map<String, Object> hit : hits) {
            assertNull("_source should be absent", hit.get("_source"));
        }
        List<String> ids = ids(hits);
        assertEquals("a", ids.get(0));
        assertTrue("diverse c ahead of near-dup b even with _source disabled", ids.indexOf("c") < ids.indexOf("b"));
        deleteKNNIndex(index);
    }

    // DIV-IT-11: explain present on each hit, mentioning the diversify/MMR selection.
    @SneakyThrows
    public void testExplain() {
        String index = "diversify_explain";
        createIndex(index, KNNEngine.FAISS);
        indexCorpus(index);

        Response response = searchKNNIndex(index, diversifyBody(0.5f, false, true), 10);
        String body = EntityUtils.toString(response.getEntity());
        assertTrue("explanation mentions diversify", body.contains("diversify"));
        deleteKNNIndex(index);
    }

    // DIV-IT-13: validation — vector_field pointing at a non-knn_vector field fails.
    @SneakyThrows
    public void testNonKnnVectorFieldRejected() {
        String index = "diversify_badfield";
        createIndex(index, KNNEngine.FAISS);
        indexCorpus(index);

        String body = "{\"retriever\":{\"diversify\":{\"vector_field\":\"title\",\"lambda\":0.5,\"window_size\":10,"
            + "\"retriever\":{\"standard\":{\"query\":{\"knn\":{\""
            + FIELD
            + "\":{\"vector\":[1.0,0.0],\"k\":10}}}}}}},\"size\":3}";
        ResponseException e = expectThrows(ResponseException.class, () -> searchKNNIndex(index, body, 10));
        assertTrue(
            "non-knn_vector field rejected",
            EntityUtils.toString(e.getResponse().getEntity()).toLowerCase().contains("knn_vector")
                || e.getResponse().getStatusLine().getStatusCode() >= 400
        );
        deleteKNNIndex(index);
    }

    // DIV-IT-3: per-leg, pre-fusion. A diversify node wraps the knn leg INSIDE a rank_fusion subtree
    // (rank_fusion[ diversify(standard(knn)), standard(match) ]). This is the §11.1 window-preserving path:
    // without preservesFusionWindow()=true the base rejects a transformer under fusion with "only allowed
    // at the top level". The test proves the per-leg placement is ACCEPTED, executes, and yields a complete
    // fused window. (The post-fusion order of the two near-duplicates is intentionally not asserted: RRF
    // blends the diversified leg with the bm25 leg, and a symmetric bm25 contribution can cancel the leg's
    // b-vs-c separation — so a specific c<b order is not a stable property of the fused output. The mirror
    // negative case, rejection without the opt-in, is DIV-IT-16 in the core TransformerRetrieverBuilderTests.)
    @SneakyThrows
    public void testPerLegUnderFusionDiversifies() {
        String index = "diversify_perleg";
        createIndex(index, KNNEngine.FAISS);
        indexCorpusWithTitles(index);

        // diversify leg: high diversity (lambda=0.1) over the knn query near a/b, nested under rank_fusion.
        String diversifyLeg = "{\"diversify\":{\"vector_field\":\""
            + FIELD
            + "\",\"lambda\":0.1,\"window_size\":10,"
            + "\"retriever\":{\"standard\":{\"query\":{\"knn\":{\""
            + FIELD
            + "\":{\"vector\":[1.0,0.0],\"k\":10}}}}}}}";
        // bm25 leg: matches all three docs on the shared title term.
        String bm25Leg = "{\"standard\":{\"query\":{\"match\":{\"title\":\"widget\"}}}}";
        String body = "{\"retriever\":{\"rank_fusion\":{\"retrievers\":["
            + diversifyLeg
            + ","
            + bm25Leg
            + "],\"rank_window_size\":10}},\"size\":3}";

        // Must be accepted (per-leg opt-in) and return the complete fused window — the diversify node ran
        // inside the fusion subtree rather than being rejected as top-level-only.
        List<String> ids = ids(search(index, body));
        assertEquals("per-leg diversify under fusion is accepted and the fused window is complete", 3, ids.size());
        assertTrue("fused result contains all corpus docs", ids.contains("a") && ids.contains("b") && ids.contains("c"));
        assertEquals("top relevance/text match a stays first after fusion", "a", ids.get(0));
        deleteKNNIndex(index);
    }
}

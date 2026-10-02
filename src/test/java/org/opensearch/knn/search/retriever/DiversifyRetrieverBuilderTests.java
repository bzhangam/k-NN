/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.retriever;

import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.SearchModule;

import java.util.Collections;

/**
 * Unit tests for {@link DiversifyRetrieverBuilder}: XContent parse (defaults + explicit), toXContent
 * round-trip, validation, and name. Nested {@code standard} child parsing resolves via the core fallback
 * retriever registry (which always registers {@code standard}).
 */
public class DiversifyRetrieverBuilderTests extends KNNTestCase {

    private NamedXContentRegistry xContentRegistry;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        SearchModule searchModule = new SearchModule(Settings.EMPTY, Collections.emptyList());
        xContentRegistry = new NamedXContentRegistry(searchModule.getNamedXContents());
    }

    @Override
    protected NamedXContentRegistry xContentRegistry() {
        return xContentRegistry;
    }

    private XContentParser parser(String json) throws Exception {
        XContentParser p = JsonXContent.jsonXContent.createParser(xContentRegistry, null, json);
        p.nextToken(); // position at START_OBJECT of the diversify body
        return p;
    }

    private static final String CHILD = "\"retriever\":{\"standard\":{\"query\":{\"match\":{\"title\":\"headphones\"}}}}";

    public void testParseDefaults() throws Exception {
        DiversifyRetrieverBuilder b = DiversifyRetrieverBuilder.fromXContent(parser("{\"vector_field\":\"embedding\"," + CHILD + "}"));
        assertEquals("diversify", b.getName());
        assertNotNull(b.getRetriever());
        // defaults applied via round-trip rendering
        String rendered = render(b);
        assertTrue(rendered, rendered.contains("\"strategy\":\"mmr\""));
        assertTrue(rendered, rendered.contains("\"lambda\":0.5"));
        assertTrue(rendered, rendered.contains("\"window_size\":100"));
        assertTrue(rendered, rendered.contains("\"vector_field\":\"embedding\""));
    }

    public void testParseExplicitFields() throws Exception {
        DiversifyRetrieverBuilder b = DiversifyRetrieverBuilder.fromXContent(
            parser(
                "{\"strategy\":\"mmr\",\"lambda\":0.2,\"window_size\":50,\"vector_field\":\"embedding\","
                    + "\"space_type\":\"l2\",\"vector_data_type\":\"float\","
                    + CHILD
                    + "}"
            )
        );
        String rendered = render(b);
        assertTrue(rendered, rendered.contains("\"lambda\":0.2"));
        assertTrue(rendered, rendered.contains("\"window_size\":50"));
        assertTrue(rendered, rendered.contains("\"space_type\":\"l2\""));
        assertTrue(rendered, rendered.contains("\"vector_data_type\":\"float\""));
    }

    public void testRoundTrip() throws Exception {
        DiversifyRetrieverBuilder b1 = DiversifyRetrieverBuilder.fromXContent(
            parser("{\"lambda\":0.3,\"window_size\":25,\"vector_field\":\"embedding\"," + CHILD + "}")
        );
        String r1 = render(b1); // {"diversify":{...}}
        DiversifyRetrieverBuilder b2 = DiversifyRetrieverBuilder.fromXContent(bodyParser(r1));
        String r2 = render(b2);
        assertEquals("render → parse → render is stable", r1, r2);
    }

    /** Parse the diversify body out of a full {@code {"diversify":{...}}} render. */
    private XContentParser bodyParser(String wrappedJson) throws Exception {
        XContentParser p = JsonXContent.jsonXContent.createParser(xContentRegistry, null, wrappedJson);
        p.nextToken(); // START_OBJECT (outer)
        p.nextToken(); // FIELD_NAME "diversify"
        p.nextToken(); // START_OBJECT (diversify body)
        return p;
    }

    public void testUnknownFieldRejected() throws Exception {
        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> DiversifyRetrieverBuilder.fromXContent(parser("{\"bogus\":1,\"vector_field\":\"embedding\"," + CHILD + "}"))
        );
        assertTrue(e.getMessage(), e.getMessage().contains("unknown field"));
    }

    // ---- validation ----

    public void testValidateMissingVectorField() {
        DiversifyRetrieverBuilder b = new DiversifyRetrieverBuilder(standardChild(), null, "mmr", 0.5f, 100, null, null);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, b::validate);
        assertTrue(e.getMessage(), e.getMessage().contains("vector_field"));
    }

    public void testValidateLambdaOutOfRange() {
        for (float bad : new float[] { -0.1f, 1.1f }) {
            DiversifyRetrieverBuilder b = new DiversifyRetrieverBuilder(standardChild(), "embedding", "mmr", bad, 100, null, null);
            IllegalArgumentException e = expectThrows(IllegalArgumentException.class, b::validate);
            assertTrue(e.getMessage(), e.getMessage().contains("lambda"));
        }
    }

    public void testValidateLambdaBoundariesAccepted() {
        for (float ok : new float[] { 0.0f, 1.0f }) {
            DiversifyRetrieverBuilder b = new DiversifyRetrieverBuilder(standardChild(), "embedding", "mmr", ok, 100, null, null);
            b.validate(); // no throw
        }
    }

    public void testValidateWindowSizeNonPositive() {
        DiversifyRetrieverBuilder b = new DiversifyRetrieverBuilder(standardChild(), "embedding", "mmr", 0.5f, 0, null, null);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, b::validate);
        assertTrue(e.getMessage(), e.getMessage().contains("window_size"));
    }

    public void testValidateUnknownStrategy() {
        DiversifyRetrieverBuilder b = new DiversifyRetrieverBuilder(standardChild(), "embedding", "cosine_mmr", 0.5f, 100, null, null);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, b::validate);
        assertTrue(e.getMessage(), e.getMessage().contains("strategy"));
    }

    public void testValidateMissingChild() {
        DiversifyRetrieverBuilder b = new DiversifyRetrieverBuilder(null, "embedding", "mmr", 0.5f, 100, null, null);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, b::validate);
        assertTrue(e.getMessage(), e.getMessage().contains("requires a [retriever] child"));
    }

    // Build a standard child without needing the registry: parse a minimal standard retriever.
    private org.opensearch.search.retriever.RetrieverBuilder standardChild() {
        try {
            XContentParser p = parser("{\"vector_field\":\"embedding\"," + CHILD + "}");
            return DiversifyRetrieverBuilder.fromXContent(p).getRetriever();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private String render(ToXContent x) throws Exception {
        XContentBuilder b = XContentFactory.jsonBuilder();
        b.startObject();
        x.toXContent(b, ToXContent.EMPTY_PARAMS);
        b.endObject();
        return b.toString();
    }
}

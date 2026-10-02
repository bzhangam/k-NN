/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.retriever;

import org.apache.lucene.search.Explanation;
import org.opensearch.action.OriginalIndices;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.opensearch.knn.search.processor.mmr.MMRExplainInfo;
import org.opensearch.knn.search.processor.mmr.MMRSelector;
import org.opensearch.knn.search.processor.mmr.MMRUtil;
import org.opensearch.knn.search.processor.mmr.MMRVectorFieldInfo;
import org.opensearch.search.fetch.subphase.FieldAndFormat;
import org.opensearch.search.retriever.LeafPreparationContext;
import org.opensearch.search.retriever.RetrieverBuilder;
import org.opensearch.search.retriever.RetrieverCandidate;
import org.opensearch.search.retriever.StandardRetrieverBuilder;
import org.opensearch.search.retriever.TransformerRetrieverBuilder;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The {@code diversify} retriever: re-ranks a single child's candidate window for relevance + diversity using
 * Maximal Marginal Relevance (MMR), so results are not near-duplicates. It is a single-child reranker
 * (extends {@link TransformerRetrieverBuilder}); the vectors driving diversity are pulled from a
 * {@code knn_vector} field via a {@code docvalue_fields} ride-along on the child leg (no second fetch).
 * <p>
 * Owned by the k-NN plugin (it depends on {@code knn_vector}, the doc-values codec, and
 * {@link KNNVectorSimilarityFunction}) and registered via the core {@code RetrieverPlugin} SPI.
 *
 * @see MMRSelector for the shared greedy selection, also used by the pipeline MMR processor.
 */
public class DiversifyRetrieverBuilder extends TransformerRetrieverBuilder {

    public static final String NAME = "diversify";

    static final String STRATEGY_FIELD = "strategy";
    static final String LAMBDA_FIELD = "lambda";
    static final String WINDOW_SIZE_FIELD = "window_size";
    static final String VECTOR_FIELD_FIELD = "vector_field";
    static final String SPACE_TYPE_FIELD = "space_type";
    static final String VECTOR_DATA_TYPE_FIELD = "vector_data_type";

    static final String MMR_STRATEGY = "mmr";
    static final float DEFAULT_LAMBDA = 0.5f;
    static final int DEFAULT_WINDOW_SIZE = 100;
    static final String DOCVALUE_FORMAT_ARRAY = "array";

    private final String strategy;
    private final float lambda;
    private final int windowSize;
    private final String vectorField;
    private final String spaceType;        // user-provided, optional (resolved from mapping/model when null)
    private final String vectorDataType;   // user-provided, optional

    // Resolved during afterChildResolved (coordinator-only; not serialized).
    private transient SpaceType resolvedSpaceType;
    private transient VectorDataType resolvedVectorDataType;
    private transient Map<String, Object> idToVector;
    private transient Map<String, MMRExplainInfo> explainInfoById;

    public DiversifyRetrieverBuilder(
        RetrieverBuilder retriever,
        String vectorField,
        String strategy,
        float lambda,
        int windowSize,
        String spaceType,
        String vectorDataType
    ) {
        super(retriever);
        this.vectorField = vectorField;
        this.strategy = strategy == null ? MMR_STRATEGY : strategy;
        this.lambda = lambda;
        this.windowSize = windowSize;
        this.spaceType = spaceType;
        this.vectorDataType = vectorDataType;
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected void validateTransformer() {
        if (vectorField == null || vectorField.isBlank()) {
            throw new IllegalArgumentException("[" + NAME + "] requires a non-empty [" + VECTOR_FIELD_FIELD + "]");
        }
        if (lambda < 0.0f || lambda > 1.0f) {
            throw new IllegalArgumentException("[" + NAME + "] [" + LAMBDA_FIELD + "] must be in [0,1], got [" + lambda + "]");
        }
        if (windowSize <= 0) {
            throw new IllegalArgumentException("[" + NAME + "] [" + WINDOW_SIZE_FIELD + "] must be > 0, got [" + windowSize + "]");
        }
        if (MMR_STRATEGY.equals(strategy) == false) {
            throw new IllegalArgumentException(
                "[" + NAME + "] only supports [" + STRATEGY_FIELD + "] = [" + MMR_STRATEGY + "], got [" + strategy + "]"
            );
        }
    }

    // A window-preserving reranker: under fusion it diversifies within the inherited window and emits the
    // same count, so it is allowed inside a fusion subtree (per-leg placement).
    @Override
    protected boolean preservesFusionWindow() {
        return true;
    }

    @Override
    public void prepareLeaves(LeafPreparationContext context) {
        // Base handles top-level-only vs window-preserving placement and propagation to the child.
        super.prepareLeaves(context);
        // Ride-along: inject the vector field as a docvalue_field onto every leaf leg under this node, so its
        // vector rides the leg's own fetch (no second search). Read back in afterChildResolved.
        for (StandardRetrieverBuilder leaf : collectLeaves()) {
            leaf.addDocvalueField(new FieldAndFormat(vectorField, DOCVALUE_FORMAT_ARRAY));
        }
    }

    @Override
    protected void afterChildResolved(Client client, String[] indices, SearchRequest original, ActionListener<Void> whenReady) {
        try {
            final SpaceType userSpaceType = spaceType == null ? null : SpaceType.getSpace(spaceType);
            final VectorDataType userVectorDataType = vectorDataType == null ? null : VectorDataType.get(vectorDataType);

            final List<IndexMetadata> localIndexMetadata = KNNClusterUtil.instance()
                .getIndexMetadataList(new OriginalIndices(indices, original == null ? null : original.indicesOptions()));

            MMRUtil.resolveKnnVectorFieldInfo(
                vectorField,
                userSpaceType,
                userVectorDataType,
                localIndexMetadata,
                client,
                ActionListener.wrap(fieldInfo -> {
                    this.resolvedSpaceType = fieldInfo.getSpaceType();
                    this.resolvedVectorDataType = fieldInfo.getVectorDataType();
                    this.idToVector = readVectorsFromWindow(fieldInfo);
                    whenReady.onResponse(null);
                }, whenReady::onFailure)
            );
        } catch (Exception e) {
            whenReady.onFailure(e);
        }
    }

    /** Decode the ride-along docvalue vector off each candidate in the window. Missing → absent (maxSim=0). */
    private Map<String, Object> readVectorsFromWindow(MMRVectorFieldInfo fieldInfo) {
        final boolean isFloat = VectorDataType.FLOAT.equals(fieldInfo.getVectorDataType());
        final Map<String, Object> vectors = new HashMap<>();
        for (RetrieverCandidate candidate : childWindow()) {
            final Object decoded = decodeVector(candidate.field(vectorField), isFloat);
            if (decoded != null) {
                vectors.put(candidate.id(), decoded);
            }
            // else: no vector for this doc → leave absent; MMRSelector treats it as incomparable (maxSim=0).
        }
        return vectors;
    }

    /**
     * Decode a doc-value field value into a {@code float[]}/{@code byte[]}. For a {@code knn_vector} with
     * {@code format=array}, {@code DocumentField.getValues()} is a single-element list whose element is the
     * whole vector — already a {@code float[]}/{@code byte[]} (native path), or a {@code List<Number>}
     * (serialized path). Returns {@code null} for a missing/empty value.
     */
    private static Object decodeVector(Object raw, boolean isFloat) {
        if (raw == null) {
            return null;
        }
        Object v = raw;
        // Unwrap a single-element wrapper list ([ vector ]).
        if (v instanceof List<?> outer) {
            if (outer.isEmpty()) {
                return null;
            }
            // If the list's first element is itself an array/list, the vector is that element.
            Object first = outer.get(0);
            if (first instanceof float[] || first instanceof byte[] || first instanceof double[] || first instanceof List) {
                v = first;
            } else {
                // The list is the component list itself (List<Number>).
                return fromNumberList(outer, isFloat);
            }
        }
        if (v instanceof float[] f) {
            if (isFloat) {
                return f;
            }
            byte[] b = new byte[f.length];
            for (int i = 0; i < f.length; i++) {
                b[i] = (byte) f[i];
            }
            return b;
        }
        if (v instanceof byte[] b) {
            if (isFloat) {
                float[] f = new float[b.length];
                for (int i = 0; i < b.length; i++) {
                    f[i] = b[i];
                }
                return f;
            }
            return b;
        }
        if (v instanceof double[] d) {
            if (isFloat) {
                float[] f = new float[d.length];
                for (int i = 0; i < d.length; i++) {
                    f[i] = (float) d[i];
                }
                return f;
            }
            byte[] bb = new byte[d.length];
            for (int i = 0; i < d.length; i++) {
                bb[i] = (byte) d[i];
            }
            return bb;
        }
        if (v instanceof List<?> components) {
            return fromNumberList(components, isFloat);
        }
        return null;
    }

    private static Object fromNumberList(List<?> values, boolean isFloat) {
        final int n = values.size();
        if (n == 0) {
            return null;
        }
        if (isFloat) {
            float[] out = new float[n];
            for (int i = 0; i < n; i++) {
                out[i] = ((Number) values.get(i)).floatValue();
            }
            return out;
        }
        byte[] out = new byte[n];
        for (int i = 0; i < n; i++) {
            out[i] = ((Number) values.get(i)).byteValue();
        }
        return out;
    }

    @Override
    protected List<RetrieverCandidate> reshape(List<RetrieverCandidate> childWindow) {
        if (childWindow.isEmpty()) {
            return childWindow;
        }
        // Bound to the window; deterministic candidate order = child position, then (index, shardId, id).
        final List<RetrieverCandidate> ordered = new ArrayList<>(childWindow);
        ordered.sort((a, b) -> {
            int c = Integer.compare(a.position(), b.position());
            if (c != 0) {
                return c;
            }
            c = a.index().compareTo(b.index());
            if (c != 0) {
                return c;
            }
            c = Integer.compare(a.shardId().id(), b.shardId().id());
            if (c != 0) {
                return c;
            }
            return a.id().compareTo(b.id());
        });

        final int limit = Math.min(windowSize, ordered.size());
        final List<RetrieverCandidate> window = ordered.subList(0, limit);

        final Map<String, RetrieverCandidate> byId = new HashMap<>(window.size());
        final List<String> orderedIds = new ArrayList<>(window.size());
        final Map<String, Float> idToScore = new HashMap<>(window.size());
        for (RetrieverCandidate c : window) {
            byId.put(c.id(), c);
            orderedIds.add(c.id());
            idToScore.put(c.id(), c.score());
        }

        // Top level: emit the request size (the executor truncates to size downstream, but MMR selects the
        // best-diversified prefix). Under fusion (window-preserving): emit the whole window so the fused
        // window count is unchanged.
        final int outputSize = effectiveWindow == LeafPreparationContext.NO_WINDOW ? orderedIds.size() : effectiveWindow;

        final float diversity = 1.0f - lambda;
        final boolean isFloat = VectorDataType.FLOAT.equals(resolvedVectorDataType);
        final KNNVectorSimilarityFunction similarity = resolvedSpaceType.getKnnVectorSimilarityFunction();

        this.explainInfoById = explain ? new HashMap<>() : null;
        final List<String> selectedIds = MMRSelector.select(
            orderedIds,
            idToVector,
            idToScore,
            similarity,
            diversity,
            outputSize,
            isFloat,
            explainInfoById
        );

        // Rebuild candidates in MMR selection order. The framework orders the final response by _score, so
        // (like pin) assign strictly-descending synthetic scores that reproduce the selection order; the
        // original child relevance is preserved in the explain payload (originalScore). Starting at the top
        // selected doc's own score and stepping down by a fixed gap keeps scores positive and ordered.
        final List<RetrieverCandidate> result = new ArrayList<>(selectedIds.size());
        final float topScore = selectedIds.isEmpty() ? 0f : idToScore.getOrDefault(selectedIds.get(0), 1f);
        final float base = Math.max(topScore, (float) selectedIds.size()); // headroom so all stay > 0
        int position = 0;
        for (String id : selectedIds) {
            RetrieverCandidate c = byId.get(id);
            float syntheticScore = base - position; // strictly descending, reproduces MMR order under score-sort
            result.add(c.withScoreAndPosition(syntheticScore, position));
            position++;
        }
        return result;
    }

    @Override
    public Explanation buildExplanation(String index, String id) {
        final Explanation childExplanation = retriever.buildExplanation(index, id);
        if (explainInfoById == null) {
            return childExplanation;
        }
        final MMRExplainInfo info = explainInfoById.get(id);
        if (info == null) {
            return childExplanation;
        }
        return Explanation.match(
            childExplanation.getValue(),
            String.format(
                Locale.ROOT,
                "%s: selected by MMR [lambda=%.4f, diversity=%.4f, max_similarity_to_selected=%.4f, mmr_score=%.4f]",
                NAME,
                lambda,
                info.getDiversity(),
                info.getMaxSimilarityToSelected(),
                info.getMmrScore()
            ),
            childExplanation
        );
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.field(STRATEGY_FIELD, strategy);
        builder.field(LAMBDA_FIELD, lambda);
        builder.field(WINDOW_SIZE_FIELD, windowSize);
        builder.field(VECTOR_FIELD_FIELD, vectorField);
        if (spaceType != null) {
            builder.field(SPACE_TYPE_FIELD, spaceType);
        }
        if (vectorDataType != null) {
            builder.field(VECTOR_DATA_TYPE_FIELD, vectorDataType);
        }
        writeChildRetriever(builder, params);
        builder.endObject();
        return builder;
    }

    public static DiversifyRetrieverBuilder fromXContent(XContentParser parser) throws IOException {
        RetrieverBuilder child = null;
        String strategy = MMR_STRATEGY;
        float lambda = DEFAULT_LAMBDA;
        int windowSize = DEFAULT_WINDOW_SIZE;
        String vectorField = null;
        String spaceType = null;
        String vectorDataType = null;

        String currentField = null;
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentField = parser.currentName();
            } else if (token == XContentParser.Token.START_OBJECT) {
                if ("retriever".equals(currentField)) {
                    child = RetrieverBuilder.parseInnerRetrieverBuilder(parser);
                } else {
                    throw new IllegalArgumentException("[" + NAME + "] unknown object field [" + currentField + "]");
                }
            } else if (token.isValue()) {
                switch (currentField) {
                    case STRATEGY_FIELD -> strategy = parser.text();
                    case LAMBDA_FIELD -> lambda = parser.floatValue();
                    case WINDOW_SIZE_FIELD -> windowSize = parser.intValue();
                    case VECTOR_FIELD_FIELD -> vectorField = parser.text();
                    case SPACE_TYPE_FIELD -> spaceType = parser.text();
                    case VECTOR_DATA_TYPE_FIELD -> vectorDataType = parser.text();
                    default -> throw new IllegalArgumentException("[" + NAME + "] unknown field [" + currentField + "]");
                }
            } else {
                throw new IllegalArgumentException("[" + NAME + "] unexpected token [" + token + "]");
            }
        }

        return new DiversifyRetrieverBuilder(child, vectorField, strategy, lambda, windowSize, spaceType, vectorDataType);
    }
}

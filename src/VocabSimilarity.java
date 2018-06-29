import edu.stanford.nlp.neural.Embedding;
import org.ejml.simple.SimpleMatrix;

import java.util.*;
import java.util.Map.Entry;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

public class VocabSimilarity {

    String vectorFile;
    Embedding embedding;

    public VocabSimilarity (String embeddingFile) {
        this.vectorFile = embeddingFile;
        this.embedding = new Embedding(vectorFile);

    }

    /**
     * calculate the cosine similarity between two vectors
     * returns a value between -1 and 1
     */
    public static double cosineSimilarity(SimpleMatrix a, SimpleMatrix b) {
        return a.dot(b) / (a.normF() * b.normF());
    }

    public Map<String, Double> getClosestWordVectors(String word, int threshold) {
        SimpleMatrix vector = this.embedding.get(word);
        Map<String, Double> similarityMap = new HashMap<>();
        if (vector == null) {
            return new HashMap<>();
        }
        for (Entry<String, SimpleMatrix> kv: this.embedding.entrySet()) {
            String k = kv.getKey();
            SimpleMatrix v = kv.getValue();
            similarityMap.putIfAbsent(k, cosineSimilarity(vector, v));
        }

        //HashMap<String, Double> sortedsims = sims.entrySet().stream() .sorted(Entry.comparingByValue()) .collect(toMap(Entry::getKey, Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        List<Entry<String, Double>> sortedSimilarity = similarityMap.entrySet().stream().sorted(Entry.comparingByValue()).collect(toList());
        Collections.reverse(sortedSimilarity);


        if (threshold <= 0) {
            return sortedSimilarity.stream().collect(toMap(Entry::getKey, Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
        }

        List<Entry<String, Double>> closest = sortedSimilarity.subList(0, threshold);

        return closest.stream().collect(toMap(Entry::getKey, Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
    }

    public static void main(String[] args) {

        VocabSimilarity similarity = new VocabSimilarity("embeddings/glove.6B.50d.txt");

       SimpleMatrix cat = similarity.embedding.get("german");

       Map<String, Double> catNeighbors = similarity.getClosestWordVectors("german", 20);
       System.out.println(catNeighbors);

    }

}

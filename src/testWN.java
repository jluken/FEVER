import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class testWN {
	public static void main(String[] args) throws IOException {
		
		Properties props;
        props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");

        // StanfordCoreNLP loads a lot of models, so you probably
        // only want to do this once per execution
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
//		
//		 
//		
//		 // look up first sense of the word "dog "
//		 String word = "ep";
//		 List<String> syns = getSynonyms(pipeline, word, POS.VERB);
//		 System . out . println (" Synset = " + syns.toString()) ;
//		 
		 String sent = "kjdasfhlkdsafhal";
		 List<String> sentLemmas = lemmatize(pipeline, sent);
		 System.out.println("sent lemmas: " + sentLemmas.toString());
		 }
	
//	public static String lemmatize(StanfordCoreNLP pipeline, String documentText)
//    {
//        List<String> lemmas = new ArrayList<String>();
//
//        // create an empty Annotation just with the given text
//        System.out.println("getting lemma");
//        Annotation document = new Annotation(documentText);
//
//        // run all Annotators on this text
//        pipeline.annotate(document);
//
//        // Iterate over all of the sentences found
//        String lemma = document.get(SentencesAnnotation.class).get(0).get(TokensAnnotation.class).get(0).get(LemmaAnnotation.class);
//        System.out.println("lemma: " + lemma);
//
//        return lemma;
//    }
	
	public static List<String> lemmatize(StanfordCoreNLP pipeline, String text)
    {
        List<String> lemmas = new ArrayList<String>();
        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        for(CoreMap sentence: sentences) {
            for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                lemmas.add(token.get(LemmaAnnotation.class));
            }
        }

        return lemmas;
    }
	
	public static List<String> getSynonyms (StanfordCoreNLP pipeline, String word, POS pos) throws IOException{
		 URL url = new URL ("file", null , "dict" ) ;
		 IDictionary dict = new Dictionary ( url ) ;
		 dict.open () ;
		 String lemma = lemmatize(pipeline, word).get(0);
		 System.out.println(lemma);
		 IIndexWord idxWord = dict.getIndexWord (lemma, pos) ;
		 System.out.println(idxWord.toString());
		 IWordID wordID = idxWord.getWordIDs().get(0) ;
		 IWord iword = dict.getWord(wordID);
		 List<String> syns = new ArrayList<String>();
		 ISynset synset = iword.getSynset();
		 for (IWord syn : synset.getWords()) {
			 syns.add(syn.getLemma().replace("_", " "));
	        }
		 return syns;
	}

}

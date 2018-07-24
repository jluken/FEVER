import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.Normalizer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;
import edu.stanford.nlp.util.CoreMap;


public class FEVER_OSU {
	
	static String claimsFileName = "shared_task_dev_public.jsonl";
	static String outputFileName = "dev_predicted_evidence.jsonl";
	static String wikiDirName = "wiki-dump";

	static int numClaimsToTest = 20;
	static int claimBatchSize = 5;
	static boolean testAll = false;
	
	static Map<String, Map<String, Object>> wikiMap;
	static Map<String, Map<String, Float>> correlationMap;
	static Map<String, ArrayList<String>> disambiguationMap;
	static Map<String, String> lowercaseMap;
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    System.out.println("Beginning document processing. Time: "+dtf.format(LocalDateTime.now()));	    
		
		StanfordCoreNLP pipeline = establishPipeline();
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(LocalDateTime.now()));	    
	    compileWikiMaps();
		System.out.println("wikiMaps compiled. Time: "+dtf.format(LocalDateTime.now()));
		
		int claimCount =0;
		try {
			Scanner claimReader = new Scanner(new FileReader(claimsFileName));
			File oldFile = new File(outputFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName, true));
			
			
			while(claimReader.hasNext() && (testAll || claimCount < numClaimsToTest)) {
				claimCount++;
				System.out.println("Finding sentences for claim "+ claimCount + ". Time: " + dtf.format(LocalDateTime.now()));
				try {
					String claimInfo = claimReader.nextLine();
					JSONObject claimJson = new JSONObject(claimInfo);
					String claim = Normalizer.normalize(claimJson.getString("claim"), Normalizer.Form.NFC);   
					int id = claimJson.getInt("id");
					System.out.println("Claim: "+claim);
					
					CoreDocument document = new CoreDocument(claim);
					pipeline.annotate(document);
					CoreSentence claimDoc = document.sentences().get(0);
					Tree constituencyTree = claimDoc.constituencyParse();
					SemanticGraph dependencyGraph = claimDoc.dependencyParse();
					String formattedClaim = formatSentence(claim);
					ArrayList<String[]> claimNE = getNamedEntities(formattedClaim, pipeline);
					System.out.println();
					Map<String, Object> documents = findDocuments(claim, dependencyGraph, constituencyTree, claimNE);
					ArrayList<Map<String, String>> primaryDocuments = (ArrayList<Map<String, String>>) documents.get("primary");
					ArrayList<String> backupDocumentKeys = (ArrayList<String>) documents.get("backup");
					List<String> primaryDocumentKeys = primaryDocuments.stream().map(doc -> doc.get("id")).collect(Collectors.toList());
					System.out.println("Primary documents found: " + primaryDocumentKeys.toString());
					
					Map<String, ArrayList<Object[]>> evidenceSentences = findSentences(pipeline, claim, dependencyGraph, constituencyTree, claimNE, primaryDocuments);
				    if (evidenceSentences.isEmpty()){
				    	System.out.println("No evidence sentences found in primary documents.");
				    	System.out.println("Backup documents: " + backupDocumentKeys.toString());
				    	ArrayList<Map<String, String>> backupDocuments = getBackupDocs(backupDocumentKeys);
				    	evidenceSentences = findSentences(pipeline, claim, dependencyGraph, constituencyTree, claimNE, backupDocuments);
				    }
				    
				    String JSONStr = evidenceToLine(id, claim, evidenceSentences);
				    writer.append(JSONStr);
				    writer.append("\n");
				}catch(Exception e){
					e.printStackTrace();
					System.out.println("Something went wrong with processeing claim " + claimCount + ". Skipping");
				    writer.append("\n");
				}
				if(claimCount % claimBatchSize == 0) {
					writer.close();
					writer = new BufferedWriter(new FileWriter(outputFileName, true));
				}
			    
			}
			claimReader.close();
			writer.close();
		}catch(Exception e) {
			e.printStackTrace();
		}

		
	}
	
	private static StanfordCoreNLP establishPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma, ner");
	    props.setProperty("coref.algorithm", "neural");
	    props.put("ner.model", "english.conll.4class.distsim.crf.ser.gz");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}
	
	@SuppressWarnings("unchecked")
	private static void compileWikiMaps() {
		try {
	         FileInputStream fileIn = new FileInputStream("wikiMap.ser");
	         ObjectInputStream in = new ObjectInputStream(fileIn);
	         System.out.println("Attempting to read wikiMap object");
	         wikiMap = (Map<String, Map<String, Object>>) in.readObject();
	         in.close();
	         fileIn.close();
	         fileIn = new FileInputStream("disambiguationMap.ser");
	         in = new ObjectInputStream(fileIn);
	         System.out.println("Attempting to read disambiguationMap object");
	         disambiguationMap = (Map<String, ArrayList<String>>) in.readObject();
	         in.close();
	         fileIn.close();
	         fileIn = new FileInputStream("lowercaseMap.ser");
	         in = new ObjectInputStream(fileIn);
	         System.out.println("Attempting to read lowercaseMap object");
	         lowercaseMap = (Map<String, String>) in.readObject();
	         in.close();
	         fileIn.close();
	    } catch (Exception e) {
	         System.out.println("Unable to read wikiMap object. Generating new object.");
	         getWikiMap(wikiDirName);
			 try {
		         FileOutputStream fileOut = new FileOutputStream("wikiMap.ser");
		         ObjectOutputStream out = new ObjectOutputStream(fileOut);
		         out.writeObject(wikiMap);
		         out.close();
		         fileOut.close();
		         fileOut = new FileOutputStream("disambiguationMap.ser");
		         out = new ObjectOutputStream(fileOut);
		         out.writeObject(disambiguationMap);
		         out.close();
		         fileOut.close();
		         fileOut = new FileOutputStream("lowercaseMap.ser");
		         out = new ObjectOutputStream(fileOut);
		         out.writeObject(lowercaseMap);
		         out.close();
		         fileOut.close();
		      } catch (IOException io) {
		         io.printStackTrace();
		      }
	    }
		
	}	
	
	private static Map<String, Object> findDocuments(String claim, SemanticGraph dependencyGraph, Tree constituencyTree, ArrayList<String[]> namedEntities){
		ArrayList<String> claimTopics = getProperTerms(claim, namedEntities);
		claimTopics.addAll(getAllTopics(claim, dependencyGraph, constituencyTree));
		System.out.println("initial all topics: " + claimTopics.toString());
		claimTopics = (ArrayList<String>) claimTopics.stream().map(topic -> StringUtils.capitalize(topic)).distinct().collect(Collectors.toList());
		claimTopics = removeSubsets(claimTopics);
		System.out.println("search: " + claimTopics.toString());
		ArrayList<Map<String, String>> primaryDocs = getDocsFromTopics(claimTopics);
		ArrayList<String> backupDocs = getBackupDocKeys(claimTopics, primaryDocs);
		Map<String, Object> allDocs = new HashMap<String, Object>();
		allDocs.put("primary", primaryDocs);
		allDocs.put("backup", backupDocs);
		return allDocs;
	}
	
	private static Map<String, ArrayList<Object[]>> findSentences(StanfordCoreNLP pipeline, String claim, SemanticGraph dependencyGraph, 
			Tree constituencyTree, ArrayList<String[]> namedEntities, ArrayList<Map<String, String>> wikis){
		Map<String, ArrayList<Object[]>> evidenceSentences = new HashMap<String, ArrayList<Object[]>>();
		String root = dependencyGraph.getFirstRoot().originalText().toLowerCase();
		for(Map<String, String> wiki : wikis) {
			ArrayList<Object[]> wikiSents = new ArrayList<Object[]>();
			String[] wikiLines = wiki.get("lines").split("\\n\\d*\\t");
			String wikiName = Normalizer.normalize(wiki.get("id"), Normalizer.Form.NFC);
			String wikiTitle = formatSentence(wikiName.replace("_", " ")).toLowerCase();
			List<String[]> nane = getNounsAndNamedEntities(wikiTitle, constituencyTree, namedEntities);
//			System.out.println("Sentences added from wiki " + wikiName + ":");
			for(int i = 0; i < wikiLines.length; i++) {
				String sentence = getSentenceTextFromWikiLines(wikiLines[i]);
//				System.out.println("sentence "+ i + ": " + sentence);
//				System.out.print("NANE: ");
//				for(String[] ne : nane) {
//					System.out.print(ne[0] + "(" + ne[1] + ")");
//				}
//				System.out.println();
				if(containsNamedEntities(sentence, claim, nane, wikiTitle, pipeline, root) || 
						containsValidRoot(sentence, root, pipeline)) {
					Object[] evidence = new Object[2];
					evidence[0] = i;
					evidence[1] = sentence;
					wikiSents.add(evidence);
//					System.out.println("added");
				}	
			}
			if(!wikiSents.isEmpty()) {
				evidenceSentences.put(wikiName, wikiSents);
			}
		}
		return evidenceSentences;
	}
	
	private static String evidenceToLine(int id, String claim, Map<String, ArrayList<Object[]>> evidenceSentences) {
		ArrayList<JSONArray> evidenceSentencesJSON = new ArrayList<JSONArray>();
		for(String wiki: evidenceSentences.keySet()) {
			ArrayList<Object[]> evidenceSets = evidenceSentences.get(wiki);
			for(Object[] evidenceSet : evidenceSets) {
				Object[] evidenceArr = {wiki, evidenceSet[0], evidenceSet[1]};
				ArrayList<JSONArray> evidenceSetJSON = new ArrayList<JSONArray>();
				evidenceSetJSON.add(new JSONArray(Arrays.asList(evidenceArr)));
				evidenceSentencesJSON.add(new JSONArray(evidenceSetJSON));
			}
		}
		JSONArray evidence = new JSONArray(evidenceSentencesJSON);
		Map<String, Object> evidenceMap = new HashMap<String, Object>();
		evidenceMap.put("id", id);
		evidenceMap.put("claim", claim);
		evidenceMap.put("evidence", evidence);
		return new JSONObject(evidenceMap).toString();
	}
	
	private static ArrayList<String> getProperTerms(String sentence, ArrayList<String[]> namedEntities){
		String[] lowerWords = {"a", "an", "the", "at", "by", "down", "for", "from", "in", "into", "like", "near", "of", "off", "on", "onto", "onto", "over", 
				"past", "to", "upon", "with", "and", "&", "as", "but", "for", "if", "nor", "once", "or", "so", "than", "that", "till", "when", "yet"};
		List<String> properTerms = new ArrayList<String>();
		String[] words = sentence.split(" ");		
		
		String properPhrase = "";
		boolean paren = false;
		for(int i = 0; i < words.length; i++) {
			String word = words[i];
			if(word.length() == 0) {
				continue;
			}
			if(Character.isUpperCase(word.charAt(0)) && properPhrase.isEmpty()){
				properPhrase = word;
			}
			else if(!properPhrase.isEmpty() && paren && (word.endsWith(")") || word.endsWith("]"))) {
				paren = false;
				properPhrase += " " + word;
				properTerms.add(properPhrase);
				properPhrase = "";
			}
			else if(!properPhrase.isEmpty() && (Character.isUpperCase(word.charAt(0)) || Character.isDigit(word.charAt(0))) || paren) {
				properPhrase += " " + word;
			}
			else if(!properPhrase.isEmpty() && Arrays.asList(lowerWords).contains(word)) {
				properTerms.add(properPhrase);
				properPhrase += " " + word;
			}
			else if(!properPhrase.isEmpty() && (word.startsWith("(") || word.startsWith("["))) {
				properTerms.add(properPhrase);
				paren = true;
				properPhrase += " " + word;
			}
			else if(!properPhrase.isEmpty()){
				properTerms.add(properPhrase);
				properPhrase = "";
			}
		}
		if(!properPhrase.isEmpty()) {
			properTerms.add(properPhrase);
		}
		List<String> dets = Arrays.asList("A", "An", "The", "There");
		List<String> withoutDets = new ArrayList<String>();
		for(String phrase : properTerms) {
			for(String det : dets) {
				if(phrase.startsWith(det + " ")) {
					withoutDets.add(phrase.substring(det.length() + 1));
				}
			}
		}
		properTerms.addAll(withoutDets);
//		List<String> fixedCase = new ArrayList<String>();
//		for(String phrase : properTerms) {
//			String fixed = phrase;
//			for(String lower : lowerWords) {
//				fixed = fixed.replace(StringUtils.capitalize(lower), lower);
//			}
//			if(!fixed.equals(phrase)) {
//				fixedCase.add(fixed);
//			}
//		}
//		properTerms.addAll(fixedCase);
		System.out.println("Possible proper terms:" + properTerms.toString());
		properTerms = properTerms.stream().map(phrase -> StringUtils.capitalize(phrase)).map(phrase -> removeEndPunct(phrase)).distinct()
				.filter(phrase -> !isInt(phrase)).filter(phrase -> !dets.contains(phrase)).filter(phrase -> isValidWiki(phrase))
				.collect(Collectors.toList());		
		properTerms = removeSubsets(properTerms);
		properTerms = removeNationalities(properTerms, namedEntities);
		System.out.println("Proper terms:" + properTerms.toString());
		return (ArrayList<String>) properTerms;
	}
	
	private static List<String> removeNationalities(List<String> terms, ArrayList<String[]> namedEntities){
		List<String> nationalities = namedEntities.stream().filter(ne -> ne[1].equals("NATIONALITY"))
				.map(ne -> ne[0]).collect(Collectors.toList());
		List<String> filtered = terms.stream().filter(term -> !nationalities.contains(term.toLowerCase())).collect(Collectors.toList());
		return filtered;
	}
	
	private static ArrayList<String> getAllTopics(String claimSentence, SemanticGraph dependencyGraph, Tree constituencyTree) {
		GrammaticalRelation[] subjectObjectRelations = {UniversalEnglishGrammaticalRelations.SUBJECT, UniversalEnglishGrammaticalRelations.CLAUSAL_PASSIVE_SUBJECT, 
	    		UniversalEnglishGrammaticalRelations.CLAUSAL_SUBJECT, UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT, 
	    		UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT, UniversalEnglishGrammaticalRelations.DIRECT_OBJECT,
	    		UniversalEnglishGrammaticalRelations.OBJECT};
		GrammaticalRelation[] dependencyRelations= {UniversalEnglishGrammaticalRelations.APPOSITIONAL_MODIFIER, UniversalEnglishGrammaticalRelations.CLAUSAL_COMPLEMENT, 
				UniversalEnglishGrammaticalRelations.XCLAUSAL_COMPLEMENT, UniversalEnglishGrammaticalRelations.POSSESSION_MODIFIER, 
				UniversalEnglishGrammaticalRelations.NAME_MODIFIER, UniversalEnglishGrammaticalRelations.NOMINAL_MODIFIER, 
				UniversalEnglishGrammaticalRelations.CASE_MARKER, UniversalEnglishGrammaticalRelations.getNmod("of")};		
	    
	    ArrayList<IndexedWord> topicList = new ArrayList<IndexedWord>();
	    Set<IndexedWord> topicWords = new HashSet<IndexedWord>();
	    topicWords.add(dependencyGraph.getFirstRoot());
		for(int dependencyRelation = 0; dependencyRelation < dependencyRelations.length; dependencyRelation++) {
			Set<IndexedWord> topicDependency = new HashSet<IndexedWord>();
			for(IndexedWord topicWord: topicWords) {
				topicDependency.addAll(getDescendentsWithReln(dependencyGraph, topicWord, dependencyRelations[dependencyRelation], 2));
			}	
    		if(!topicDependency.isEmpty()) {
    			topicList.addAll(topicDependency);
    		}
	    }
		
	    for(int relation = 0; relation < subjectObjectRelations.length; relation++) {
	    	topicWords = getDescendentsWithReln(dependencyGraph, dependencyGraph.getFirstRoot(), subjectObjectRelations[relation], 2);
	    	if(!topicWords.isEmpty()) {
    			topicList.addAll(topicWords);
    			for(int dependencyRelation = 0; dependencyRelation < dependencyRelations.length; dependencyRelation++) {
    				Set<IndexedWord> topicDependency = new HashSet<IndexedWord>();
    				for(IndexedWord topicWord: topicWords) {
    					topicDependency.addAll(getDescendentsWithReln(dependencyGraph, topicWord, dependencyRelations[dependencyRelation], 2));
    				}
    				if(!topicDependency.isEmpty()) {
    	    			topicList.addAll(topicDependency);
    	    		}
    		    }
    		}
	    }

	    ArrayList<String> topicPhrases = new ArrayList<String>();
	    for(int h = 0; h < topicList.size(); h++) {
	    	IndexedWord topicWord = topicList.get(h);
	    	String compoundPhrase = getCompoundPhrase(topicWord, dependencyGraph);
	    	if(compoundPhrase != null && isValidWiki(compoundPhrase)) {
	    		topicPhrases.add(compoundPhrase);
	    	}  	
	    	ArrayList<String> nounPhrases = getNounPhrases(claimSentence, topicWord, constituencyTree);
		    topicPhrases.addAll(nounPhrases);
	    }
	    
	    topicPhrases = (ArrayList<String>) topicPhrases.stream().distinct().collect(Collectors.toList());
		return topicPhrases;
	}
	
	private static Set<IndexedWord> getDescendentsWithReln(SemanticGraph dependencyTree, IndexedWord vertex, GrammaticalRelation reln, int layer) {
		Set<IndexedWord> descendents = dependencyTree.getChildrenWithReln(vertex, reln);
		Set<IndexedWord> children = dependencyTree.getChildren(vertex);
		for(IndexedWord child: children) {
			if(layer > 0) {
				descendents.addAll(getDescendentsWithReln(dependencyTree, child, reln, layer-1));
			}
		}
		
		return descendents;
	}
	
	private static String getCompoundPhrase(IndexedWord topic, SemanticGraph dependencies) {
		Set<IndexedWord> compoundWordSet = dependencies.getChildrenWithRelns(topic, java.util.Collections.singleton(UniversalEnglishGrammaticalRelations.COMPOUND_MODIFIER));
	    String compoundPhrase = null;
		if(!compoundWordSet.isEmpty()) {	
    		ArrayList<IndexedWord> compoundWordList = new ArrayList<IndexedWord>();
	    	compoundWordList.add(topic);
	    	for (IndexedWord compound : compoundWordSet) {
	            int phraseIndex = 0;
	            while(phraseIndex < compoundWordList.size() && compound.index() > compoundWordList.get(phraseIndex).index()) {
	            	phraseIndex++;
	            }
	            compoundWordList.add(phraseIndex, compound);
	         }
	    	compoundPhrase = compoundWordList.get(0).originalText();
	    	for(int i = 1; i < compoundWordList.size(); i++) {
	    		compoundPhrase += " ";
	    		compoundPhrase += compoundWordList.get(i).originalText();
	    	}
	    }
		
		return compoundPhrase;
	}
	
	private static ArrayList<String> getNounPhrases(String claim, IndexedWord topic, Tree constituencyTree) {
		ArrayList<String> nounPhrases = new ArrayList<String>();
		List<Tree> leaves = constituencyTree.getLeaves();
	    Tree topicTree = leaves.get(topic.index()-1);

	    //return ArrayList of topic phrases, from broadest to most narrow
	    ArrayList<Tree> topicTreeList = new ArrayList<Tree>();
	    String[] nounsAndPhrases = {"NN", "NNS", "NNP", "NNPS", "NP"};
	    boolean onlyNouns = true;
	    boolean added = false;
	    while(topicTree.parent(constituencyTree) != null) {
	    	topicTree = topicTree.parent(constituencyTree);
	    	if(!Arrays.asList(nounsAndPhrases).contains(topicTree.label().toString())) {
	    		onlyNouns = false;
	    	}
	    	if(Arrays.asList(nounsAndPhrases).contains(topicTree.label().toString()) && ((onlyNouns && !added) || topicTree.getLeaves().size() > 1)) {
	    		topicTreeList.add(0, topicTree.deepCopy());
	    		added = true;
	    	}
	    }
	    
	    
	    //only return the most "complete" valid phrases
	    int i = 0;
	    while (i < topicTreeList.size() && nounPhrases.isEmpty()) {
		    List<Tree> topicWords = topicTreeList.get(i).getLeaves();
		    String topicPhrase  = topicWords.get(0).toString();	    
		    for(int j = 1; j < topicWords.size(); j++) {
		    	boolean endsInPossessive = (j == topicWords.size()-1) && topicWords.get(j).toString().equals("\'s");
		    	if(!endsInPossessive) {
			    	topicPhrase += " ";
			    	topicPhrase += topicWords.get(j).toString();
		    	}
		    }
		    
		    
		    if(topicPhrase.startsWith("The ") || topicPhrase.startsWith("A ") || topicPhrase.startsWith("An ")) { 
		    	boolean proper = !claim.startsWith(topicPhrase);
		    	String nonDetPhrase;
		    	if(topicPhrase.startsWith("The ")) {
		    		nonDetPhrase = topicPhrase.substring(4);
		    	}else if(topicPhrase.startsWith("An ")){
		    		nonDetPhrase = topicPhrase.substring(3);
		    	}else {
		    		nonDetPhrase = topicPhrase.substring(2);
		    	}
		    	
		    	if(proper) {
		    		if(isValidWiki(topicPhrase)) {
				    	nounPhrases.add(topicPhrase);
				    }
		    	} else {
		    		if(isValidWiki(topicPhrase)) {
				    	nounPhrases.add(topicPhrase);
				    }
		    		if(isValidWiki(nonDetPhrase)) {
			    		nounPhrases.add(nonDetPhrase);
			    	}
		    	}
		    }
		    else if(topicPhrase.startsWith("the ") || topicPhrase.startsWith("a ") || topicPhrase.startsWith("an ")) { 
		    	String nonDetPhrase;
		    	if(topicPhrase.startsWith("the ")) {
		    		nonDetPhrase = topicPhrase.substring(4);
		    	}else if(topicPhrase.startsWith("an ")){
		    		nonDetPhrase = topicPhrase.substring(3);
		    	}else {
		    		nonDetPhrase = topicPhrase.substring(2);
		    	}
		    	if(isValidWiki(nonDetPhrase)) {
		    		nounPhrases.add(nonDetPhrase);
		    	}
		    }
		    else if(isValidWiki(topicPhrase)) {
		    	nounPhrases.add(topicPhrase);
		    }
		    i++;
	    }
	    
	    return nounPhrases;
	}
	
	private static List<String> lemmatize(StanfordCoreNLP pipeline, String text) {
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
	
	private static List<String> getSynonyms (StanfordCoreNLP pipeline, String word, POS pos){
		List<String> syns = new ArrayList<String>();
		String lemma = word;
		try {
			 URL url = new URL ("file", null , "dict" ) ;
			 IDictionary dict = new Dictionary ( url ) ;
			 dict.open () ;
			 lemma = lemmatize(pipeline, word).get(0);
			 IIndexWord idxWord = dict.getIndexWord (lemma, pos) ;
			 IWordID wordID = idxWord.getWordIDs().get(0) ;
			 IWord iword = dict.getWord(wordID);
			 ISynset synset = iword.getSynset();
			 for (IWord syn : synset.getWords()) {
				 syns.add(syn.getLemma().replace("_", " "));
		        }
		}catch(Exception e){
			syns.add(lemma);
		}
		 return syns;
	}
	
	private static String removeEndPunct(String phrase) {
		String removed = phrase;
		String[] punct = {"!", "?", ".",",", ";", ":", "'s", "'"};
		for (String mark: punct) {
			if (phrase.endsWith(mark)) {
				removed = phrase.substring(0, phrase.length() - mark.length());
		    }
		}
		return removed;
	}
	
	private static ArrayList<String> removeSubsets(List<String> set) {
		ArrayList<String> filtered = new ArrayList<String>();
		for (String subStr: set) {
			boolean unique = true;
			for (String str: set) {
				if(str.contains(subStr) && !str.equals(subStr) && !isDetSubstring(subStr, str)) {
					unique = false;
				}
		    }
			if(unique) {
				filtered.add(subStr);
			}
		}
		return filtered;
	}
	
	private static boolean isDetSubstring(String subStr, String str) {
		String[] dets = {"The ", "An ", "A ", "the ", "an ", "a "};
		boolean isDetSub = false;
		for(String det : dets) {
			if(str.equals(det+subStr)) {
				isDetSub = true;
			}
		}
		return isDetSub;	
	}
	
	private static boolean isValidWiki(String title) {
		boolean valid = false;
		String wikiKey = title.replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace("]", "-RSB-").replace("[", "-LSB-");
		wikiKey = StringUtils.capitalize(wikiKey);
		if(wikiMap.containsKey(wikiKey) || disambiguationMap.containsKey(wikiKey) || lowercaseMap.containsKey(wikiKey.toLowerCase())) {
			valid = true;
	    }
		return valid;
	}
	
	private static String formatWiki(String title) {
		String wikiKey = title.replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace("]", "-RSB-").replace("[", "-LSB-");
		wikiKey = StringUtils.capitalize(wikiKey);
		if(!wikiMap.containsKey(wikiKey) && !disambiguationMap.containsKey(wikiKey) && lowercaseMap.containsKey(wikiKey.toLowerCase())) {
			wikiKey = lowercaseMap.get(wikiKey.toLowerCase());
		}
		return wikiKey;
	}
	
	private static boolean isInt(String str) {
		boolean isInt = true;
		try {
			Integer.parseInt(str);
	    } catch (NumberFormatException e) {
	        isInt = false;
	    }
		return isInt;
	}
	
	private static String getSentenceTextFromWikiLines(String sentenceInfo) {
		String sentence;
		String[] tabs = sentenceInfo.split("\\t");
		if(tabs[0].equals("0")) {
			sentence = tabs[1];
		} else {
			sentence = tabs[0];
		}
		return sentence;
	}
	
	private static String formatSentence(String sentence) {
		String newSent = sentence.replace(",", " ,").replace(".", " .").replace(";", " ;").replace(":", " :").replace("'s", " 's").replace("' ", " ' ");
		newSent = newSent.replace("-LRB-", "-LRB- ").replace("-RRB-", " -RRB-").replace("-RSB-", " -RSB-").replace("-LSB-", " -LSB-");
		newSent = newSent.replace("(", "-LRB- ").replace(")", " -RRB-").replace("]", " -RSB-").replace("[", " -LSB-");
		return newSent;
	}
	
	private static boolean isVerb(String word, StanfordCoreNLP pipeline) {
		CoreDocument document = new CoreDocument(word);
	    pipeline.annotate(document);
	    CoreSentence claimDoc = document.sentences().get(0);
		Tree constituencyTree = claimDoc.constituencyParse();
		List<Tree> leaves = constituencyTree.getLeaves();
		Tree wordTree = leaves.get(0);
		Tree type = wordTree.parent(constituencyTree);
		String[] verbTags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};
		boolean verb = false;
		if(Arrays.asList(verbTags).contains(type.label().toString())) {
			verb = true;
		} 
		return verb;
		
		
	}

	private static ArrayList<String> getNouns(String wikiTitle, Tree constituencyTree) {
		String[] ignoredNouns = {"something", "anything", "thing", "there", "name", "part"};
		List<String> ignoredNounsList = Arrays.asList(ignoredNouns);
		List<Tree> leaves = constituencyTree.getLeaves();
	    int wordNum = 0;

	    ArrayList<String> wordList = new ArrayList<String>();
	    String[] nounsTags = {"NN", "NNS", "NNP", "NNPS", "NP"};
	    while(wordNum < leaves.size()) {
	    	Tree word = leaves.get(wordNum);
	    	Tree type = word.parent(constituencyTree);
	    	String wordStr = word.toString();
	    	if(Arrays.asList(nounsTags).contains(type.label().toString()) && !wikiTitle.contains(wordStr.toLowerCase()) 
	    			&& !ignoredNounsList.contains(wordStr)) {
	    		wordList.add(wordStr.toLowerCase());
	    	} 
	    	wordNum++;
	    }
    
	    return wordList;
	}

	private static ArrayList<String[]> getNounsAndNamedEntities(String wikiTitle,  Tree constituencyTree, List<String[]> namedEntities){
		ArrayList<String> nouns = getNouns(wikiTitle, constituencyTree);
	    List<String[]> filteredNamedEntities = namedEntities.stream().filter(arr -> !wikiTitle.contains(arr[0]))
				.collect(Collectors.toList());
			
	    ArrayList<String[]> nounsAndNamedEntities = new ArrayList<String[]>(filteredNamedEntities);
		for(String noun: nouns) {
			boolean contained = false;
			for(String[] namedEntity: filteredNamedEntities) {
				if(namedEntity[0].contains(noun)) {
					contained = true;
				}
			}
			if(!contained) {
				String[] nounEntry = {noun, "NOUN"};
				nounsAndNamedEntities.add(nounEntry);
			}
		}
		
		return nounsAndNamedEntities;
	}

	private static boolean containsNamedEntities(String evidenceSentence, String claim, List<String[]> claimNE, String wikiTitle, StanfordCoreNLP pipeline, String root) {
		Map<String, String> altMap = new HashMap<String, String>();
		altMap.put("NATIONALITY", "COUNTRY");
		altMap.put("COUNTRY", "NATIONALITY");
		altMap.put("DATE", "SET");
		altMap.put("SET", "DATE");
		
		boolean validSentence = true;
		if(evidenceSentence.isEmpty() || claimNE.isEmpty()) {
			return false;
		}

		String[] entityToBeSwapped = {null, null};
		for(String[] ne: claimNE) {
			if(!evidenceSentence.toLowerCase().contains(ne[0]) && entityToBeSwapped[1] == null) {
				entityToBeSwapped = ne.clone();
			}
			else if(!evidenceSentence.toLowerCase().contains(ne[0])) {
//				System.out.println("double missing");
				validSentence = false;
			}	
		}
		if(validSentence && entityToBeSwapped[1] == "NOUN" && claimNE.size() < 3) {
			//both sentences need to be noun complements, have at least 2 other matching entities
			//or the evidence needs to have a synonym of the missing noun
			if(!isNounComplement(claim) || !isNounComplement(evidenceSentence)) {
				validSentence = containsSynonym(entityToBeSwapped[0], POS.NOUN, evidenceSentence, pipeline);
				if(!validSentence) {
//					System.out.println("noun has no synonym");
				}
			}
		}	
//		if(validSentence && entityToBeSwapped[1] == "NOUN" && claimNE.size() >= 3) {
//			System.out.println("excused because of matching");
//		}
		
		List<String[]> evidenceEntities = null;
		if(validSentence && entityToBeSwapped[1] != "NOUN" && entityToBeSwapped[1] != null) {
			if(isVerb(root, pipeline) && !(evidenceSentence.contains(root) || containsSynonym(root, POS.VERB, evidenceSentence, pipeline))) {
				//if the evidence is missing a root verb and a named entity, then the evidence needs to have a synonym of the missing root
				validSentence = false;
//				System.out.println("missing root verb");
			}
			else {
				//otherwise the entity can be swapped out for one of the same type
				evidenceEntities = getNamedEntities(evidenceSentence, pipeline).stream().filter(arr -> !wikiTitle.contains(arr[0]))
						.collect(Collectors.toList());
//				System.out.print("sentence entities: ");
//				for(String[] ne : evidenceEntities) {
//					System.out.print(ne[0] + "(" + ne[1] + ")");
//				}
//				System.out.println();
				validSentence = false;
				String alternative = altMap.get(entityToBeSwapped[1]);
				for(String[] ne: evidenceEntities) {
					if(ne[1].equals(entityToBeSwapped[1]) || (alternative != null && alternative.equals(ne[1]))) {
						validSentence = true;
					}
				}
			}
		}

		if(!validSentence &&  evidenceSentence.contains("-LRB-") && 
				(claim.contains("born") || claim.contains("died") || claim.contains("dead"))) {
			//if the sentence relates to birth or death, we can check to see if the sentence has wikipedia-formatted birth/death info 
			if(evidenceEntities == null) {
				evidenceEntities = getNamedEntities(evidenceSentence, pipeline).stream().filter(arr -> !wikiTitle.contains(arr[0]))
					.collect(Collectors.toList());
			}
			validSentence = wikiBirthDeath(claim, evidenceEntities, evidenceSentence);
		}

		return validSentence;
	}
	
	private static boolean containsSynonym(String word, POS pos, String sentence, StanfordCoreNLP pipeline) {
		boolean contains = false;
		String[] ignoredLemmas = {"have", "do", "be"};
		List<String> sentLemmas = lemmatize(pipeline, sentence);
		List<String> syns = getSynonyms(pipeline, word, pos);
		for(String syn : syns) {
			if(sentLemmas.contains(syn) && !Arrays.asList(ignoredLemmas).contains(syn)) {
				contains = true;
				System.out.println("SYN: ");
				System.out.println("sentence lemmas: " + sentLemmas);
				System.out.println("syn: " + syn);
			}
		}
		return contains;
	}
	
	private static boolean isNounComplement(String sentence) {
		String[] bes = {" is", " was"};
		String[] adverbs = {" ", " only ", " always ", " never ", " not ", };
		String[] dets = {"a ", "an ", "the "};
		for(String verb : bes) {
			for(String adverb : adverbs) {
				for(String det : dets) {
					String ncTag = verb + adverb + det;
					if(sentence.contains(ncTag)) {
						return true;
					}
				}
			}
		}
		return false;
	}
	
	private static boolean containsValidRoot(String sentence, String root, StanfordCoreNLP pipeline) {
		String[] isWords = {"is", "was", "be", "are", "were", "has", "had", "have"};
		String rootAlone = " " + root + " ";
		boolean validRoot = false;
		if(isVerb(root, pipeline) && !Arrays.asList(isWords).contains(root) && 
				(sentence.toLowerCase().contains(rootAlone) || sentence.toLowerCase().startsWith(root))) {
			validRoot = true;
		}
		if(validRoot) {
			System.out.println("valid root: " + root);
		}
		return validRoot;
	}
	
	private static boolean wikiBirthDeath(String claim, List<String[]> SentNameEntities, String sentence) {
		boolean edgeCase = false;
		List<String> dates = SentNameEntities.stream().filter(ne -> ne[1].equals("DATE")).map(ne -> ne[0]).collect(Collectors.toList());
		for(String date : dates) {
			boolean parenDate = Pattern.compile("lrb(.*)" + date + "(.*)rrb").matcher(sentence.toLowerCase()).find();
			if(parenDate) {
				edgeCase = true;
			}
		}
		
		return edgeCase;
	}
	
	private static ArrayList<String[]> getNamedEntities(String sentence, StanfordCoreNLP pipeline){

		ArrayList<CoreLabel> tokens = getSentenceTokens(sentence, pipeline);
		ArrayList<String[]> namedEntities = new ArrayList<String[]>();
		String neTag = "";
		String namedEntity = "";
		boolean neActive = false;
		for(int i = 0; i < tokens.size(); i++) {
			String ne = tokens.get(i).get(NamedEntityTagAnnotation.class);
			if(!neActive && !ne.equals("O")) {
				neActive = true;
				neTag = ne;
				namedEntity += tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && ne.equals(neTag)) {
				namedEntity += " " + tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && !ne.equals(neTag) && !ne.equals("O")) {
				String[] neInfo = {namedEntity.toLowerCase(), neTag};
				namedEntities.add(neInfo);		
				namedEntity = "";
				neTag = ne;
				namedEntity = tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && ne.equals("O")) {
				neActive = false;
				String[] neInfo = {namedEntity.toLowerCase(), neTag};
				namedEntities.add(neInfo);		
				namedEntity = "";
				neTag = "";
			}
		}
		if(neActive) {
			String[] neInfo = {namedEntity.toLowerCase(), neTag};
			namedEntities.add(neInfo);
		}
		return namedEntities;
   }

	private static ArrayList<CoreLabel> getSentenceTokens(String sentence, StanfordCoreNLP pipeline){
		Annotation document = new Annotation(sentence);
	    pipeline.annotate(document);
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    ArrayList<CoreLabel> tokens = new ArrayList<CoreLabel>();
	    if(sentences.isEmpty()) {
	    	return tokens;
	    }
	    CoreMap tokenSen = sentences.get(0);
	    
	    
	    for (CoreLabel token: tokenSen.get(TokensAnnotation.class)) {
	        tokens.add(token);
	      }
	    return tokens;
	}
	
	private static ArrayList<Map<String, String>> getDocsFromTopics(ArrayList<String> possibleTopics) {
		ArrayList<Map<String, String>> wikiDocs = new ArrayList<Map<String, String>>();
		for(String topic: possibleTopics) {
			String urlTitle = formatWiki(topic);
			System.out.println("urlTitle: " + urlTitle);
			Map<String, String> wikiDoc = new HashMap<String, String>();
			boolean emptyDisam = false;
			if (!topic.isEmpty() && wikiMap.containsKey(urlTitle)){
				Map<String, Object> fileInfo = wikiMap.get(urlTitle);
//				String wikiListName = wikiDirName + "/" + fileInfo.get("fileName");
				Path wikiDirPath = Paths.get(wikiDirName);
				Path wikiListPath =  wikiDirPath.resolve(Paths.get((String)fileInfo.get("fileName")));
				String wikiListName = wikiListPath.toString();

				try {
					BufferedReader reader = new BufferedReader(new FileReader(wikiListName));
					Long byteOffset = (Long) fileInfo.get("offset");
					reader.skip(byteOffset);
				    String wikiEntry = reader.readLine();
				    JSONObject wikiJson = new JSONObject(wikiEntry);
				    if (wikiJson.getString("text").toLowerCase().contains((topic.toLowerCase()+" may refer to : ")) ||
				    		wikiJson.getString("text").toLowerCase().contains((topic.toLowerCase()+" may also refer to : "))) {
			    		ArrayList<Map<String, String>> disambiguationChildren = findDisambiguationChildren(wikiJson);
			    		wikiDocs.addAll(disambiguationChildren);
			    		if(disambiguationChildren.isEmpty()) {
			    			emptyDisam = true;
			    		}
			    	}
				    else {
					    wikiDoc.put("id", wikiJson.getString("id"));
					    wikiDoc.put("text", wikiJson.getString("text"));
					    wikiDoc.put("lines", wikiJson.getString("lines"));
					    wikiDocs.add(wikiDoc);
				    }
				    reader.close();
				}
				
				catch (FileNotFoundException e) {
					System.out.println("Could not open file  "+fileInfo.get("fileName"));
					e.printStackTrace();
				} catch (JSONException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
				
			}
			
			if (!topic.isEmpty() && disambiguationMap.containsKey(urlTitle)){
				if(emptyDisam) {
					ArrayList<String> disambiguations = disambiguationMap.get(urlTitle);
					ArrayList<String> unchecked = new ArrayList<String>(disambiguations);
					for(Map<String, String> wiki: wikiDocs) {
						if(disambiguations.contains(wiki.get("id"))) {
							unchecked.remove(wiki.get("id"));
						}
					}
					wikiDocs.addAll(getDocsFromTopics(unchecked));
				}
			}
		}
		return wikiDocs;
	}
	
	private static ArrayList<Map<String, String>> getBackupDocs(ArrayList<String> backupDocumentKeys){
		ArrayList<Map<String, String>> backupDocs = new ArrayList<Map<String, String>>();
		for(String key : backupDocumentKeys) {
			try {
				Map<String, String> backupDoc = new HashMap<String, String>();
				Map<String, Object> fileInfo = wikiMap.get(key);
//				String wikiListName = wikiDirName + "\\" + fileInfo.get("fileName");
                Path wikiDirPath = Paths.get(wikiDirName);
                Path wikiListPath =  wikiDirPath.resolve(Paths.get((String)fileInfo.get("fileName")));
                String wikiListName = wikiListPath.toString();

				BufferedReader reader = new BufferedReader(new FileReader(wikiListName));
				Long byteOffset = (Long) fileInfo.get("offset");
				reader.skip(byteOffset);
			    String wikiEntry = reader.readLine();
			    JSONObject wikiJson = new JSONObject(wikiEntry);
			    backupDoc.put("id", wikiJson.getString("id"));
			    backupDoc.put("text", wikiJson.getString("text"));
			    backupDoc.put("lines", wikiJson.getString("lines"));
			    backupDocs.add(backupDoc);
			    reader.close();
			} catch (FileNotFoundException e) {
				System.out.println("Could not open file  "+ key);
				e.printStackTrace();
			} catch (JSONException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return backupDocs;
	}
	
	private static ArrayList<String> getBackupDocKeys(ArrayList<String> possibleTopics, ArrayList<Map<String, String>> existingDocs) {
		ArrayList<String> backupDocs = new ArrayList<String>();
		for(String topic: possibleTopics) {
			String urlTitle = StringUtils.capitalize(topic.replace(' ', '_').replace("(", "-LRB-").replace(")", "-RRB-"));
			if (!topic.isEmpty() && disambiguationMap.containsKey(urlTitle)){
				backupDocs.addAll(disambiguationMap.get(urlTitle));
				for(Map<String, String> wiki: existingDocs) {
					if(backupDocs.contains(wiki.get("id"))) {
						backupDocs.remove(wiki.get("id"));
					}
				}
			}
		
		}
		return backupDocs;
	}
	
	private static ArrayList<Map<String, String>> findDisambiguationChildren(JSONObject disambiguation){
		ArrayList<Map<String, String>> disambiguationChildren = new ArrayList<Map<String, String>>();
		try {
			String lines = disambiguation.getString("lines");
			String[] entries = lines.split("[\\n[\\d+]\\t]+");
			ArrayList<String> topics = new ArrayList<String>();
			for(int i = 1; i < entries.length; i++) {
				String[] tabs = entries[i].split("\\t");
				String wiki = tabs.length > 1 ? tabs[1] : null;
				if(wiki != null && !wiki.replace(' ', '_').replace("(", "-LRB-").replace(")", "-RRB-").equals(disambiguation.getString("id"))) {
					topics.add(wiki);
				}
			}
			disambiguationChildren = getDocsFromTopics(topics);
		} catch (JSONException e) {
			e.printStackTrace();
		}
		
		return disambiguationChildren;
	}
	
	private static void getWikiMap(String wikiDirName) {
		wikiMap = new HashMap<String, Map <String, Object>>();
		disambiguationMap = new HashMap<String, ArrayList<String>>();
		lowercaseMap = new HashMap<String, String>();
		
		File wikiDir = new File(wikiDirName);
		File[] wikiEntries = wikiDir.listFiles();
		int filesProcessed = 0;
		if (wikiEntries != null){
			for (File wikiEntryList : wikiEntries) {
				try {
					System.out.print("*");
					Scanner s = new Scanner(wikiEntryList);
					long byteOffset = 0;
					while(s.hasNextLine()) {
						String wikiEntry = Normalizer.normalize(s.nextLine(), Normalizer.Form.NFC);
					    JSONObject wikiJson = new JSONObject(wikiEntry);
					    String id = wikiJson.getString("id");
					    if(!id.isEmpty()) {
					    	String fileName = wikiEntryList.getName();
					    	Map<String, Object> docLocation = new HashMap<String, Object>();
					    	docLocation.put("fileName", fileName);
					    	docLocation.put("offset", byteOffset);
							wikiMap.put(id, docLocation);
							lowercaseMap.put(id.toLowerCase(), id);
							
							int paren = id.indexOf("-LRB-");
							if(paren > 0 && !id.contains("disambiguation")) {
								String base = id.substring(0, paren-1);
								ArrayList<String> disambiguationChildren = new ArrayList<String>();
								if(disambiguationMap.containsKey(base)) {
									disambiguationChildren = disambiguationMap.get(base);
									disambiguationChildren.add(id);
									disambiguationMap.put(base, disambiguationChildren);
								}
								else {
									disambiguationChildren.add(id);
									disambiguationMap.put(base, disambiguationChildren);
								}
							}
					    }
					    byteOffset += wikiEntry.getBytes().length;
						byteOffset += 1;
					}
					filesProcessed++;
					if(filesProcessed % 10 == 0 || filesProcessed == 109) {
						System.out.println("\nWiki processing "+filesProcessed+"/109 done.");
					}
						
					s.close();
				} catch (FileNotFoundException e) {
					System.out.println("Could not open file  "+wikiEntryList.getName());
					e.printStackTrace();
				} catch (JSONException e) {
					e.printStackTrace();
				}
			}
		}
	}
	

}

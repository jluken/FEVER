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
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
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
	static String outputFileName = "claim_sentences.jsonl";
	static String wikiDirName = "wiki-dump";
	static int numClaimsToTest = 10;
	
	static Map<String, Map<String, Object>> wikiMap;
	static Map<String, Map<String, Float>> correlationMap;
	static Map<String, ArrayList<String>> disambiguationMap;
	
	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    System.out.println("Beginning document processing. Time: "+dtf.format(LocalDateTime.now()));	    
		
		StanfordCoreNLP pipeline = establishPipeline();
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(LocalDateTime.now()));	    
	    compileWikiMaps();
		System.out.println("wikiMaps compiled. Time: "+dtf.format(LocalDateTime.now()));
		compileCorrelationMap();
		System.out.println("correlationMap compiled. Time: "+dtf.format(LocalDateTime.now()));
		
		int claimCount =0;
		try {
			Scanner claimReader = new Scanner(new FileReader(claimsFileName));
			File oldFile = new File(outputFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName, true));
			
			
			while(claimReader.hasNext() && claimCount < numClaimsToTest) {
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
					String root = dependencyGraph.getFirstRoot().originalText().toLowerCase();
					String formattedClaim = formatSentence(claim);
					ArrayList<String[]> claimNE = getNamedEntities(formattedClaim, pipeline);
					
					Map<String, Object> documents = findDocuments(claim, dependencyGraph, constituencyTree);
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
	
	private static void compileWikiMaps() {
		
         getWikiMap(wikiDirName);
		 try {
	         FileOutputStream fileOut = new FileOutputStream("wikiMap.ser");
	         ObjectOutputStream out = new ObjectOutputStream(fileOut);
	         out.writeObject(wikiMap);
	         out.close();
	         fileOut.close();
	      } catch (IOException io) {
	         io.printStackTrace();
	      }
		
	}	
	
	private static void compileCorrelationMap() {
		try {
		     FileInputStream fileIn = new FileInputStream("rootCorrelations.ser");
		     ObjectInputStream in = new ObjectInputStream(fileIn);
		     correlationMap = (Map<String, Map<String, Float>>) in.readObject();
		     in.close();
		     fileIn.close();
		  } catch (Exception e) {
			  e.printStackTrace();
			  correlationMap = new HashMap<String, Map<String, Float>>();
		  } 

	}
	
	private static Map<String, Object> findDocuments(String claim, SemanticGraph dependencyGraph, Tree constituencyTree){
		ArrayList<String> claimTopics = getProperTerms(claim);
		claimTopics.addAll(getAllTopics(claim, dependencyGraph, constituencyTree));
		claimTopics = (ArrayList<String>) claimTopics.stream().distinct().collect(Collectors.toList());
		claimTopics = removeSubsets(claimTopics);
		ArrayList<Map<String, String>> primaryDocs = getDocsFromTopics(claimTopics);
		ArrayList<String> backupDocs = getBackupDocs(claimTopics, primaryDocs);
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
			System.out.println("Sentences from wiki " + wikiName + ":");
			for(int i = 0; i < wikiLines.length; i++) {
				String sentence = getSentenceTextFromWikiLines(wikiLines[i]);
				if(containsNamedEntities(sentence, claim, namedEntities, wikiTitle, pipeline, root) ||
						 containsCorrelatedWord(sentence, root, wikiTitle) ||
						 containsValidRoot(sentence, root, pipeline)) {
					Object[] evidence = new Object[2];
					evidence[0] = i;
					evidence[1] = sentence;
					wikiSents.add(evidence);
					System.out.println("Added sentence: " + sentence);
				}	
			}
			if(!wikiSents.isEmpty()) {
				evidenceSentences.put(wikiName, wikiSents);
			}
		}
		return evidenceSentences;
	}
	
	private static String evidenceToLine(int id, String claim, Map<String, ArrayList<Object[]>> evidenceSentences) {
		ArrayList<JSONArray> evidenceSetJSON = new ArrayList<JSONArray>();
		for(String wiki: evidenceSentences.keySet()) {
			ArrayList<Object[]> evidenceSets = evidenceSentences.get(wiki);
			for(Object[] evidenceSet : evidenceSets) {
				Object[] evidenceArr = {wiki, evidenceSet[0], evidenceSet[1]};
				evidenceSetJSON.add(new JSONArray(Arrays.asList(evidenceArr)));
			}
		}
		JSONArray evidence = new JSONArray(Arrays.asList(evidenceSetJSON));
		Map<String, Object> evidenceMap = new HashMap<String, Object>();
		evidenceMap.put("id", id);
		evidenceMap.put("claim", claim);
		evidenceMap.put("evidence", evidence);
		return new JSONObject(evidenceMap).toString();
	}
	
	private static ArrayList<String> getProperTerms(String sentence){
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
			else if(!properPhrase.isEmpty() && !Character.isLowerCase(word.charAt(0)) || paren) {
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
			else if(!properPhrase.isEmpty() && (word.endsWith(")") || word.endsWith("]"))) {
				paren = false;
				properPhrase += word;
				properTerms.add(properPhrase);
				properPhrase = "";
			}
			else if(!properPhrase.isEmpty()){
				properTerms.add(properPhrase);
				properPhrase = "";
			}
		}
		if(!properPhrase.isEmpty()) {
			properTerms.add(properPhrase);
		}
		System.out.println("Possible proper terms:" + properTerms.toString());
		List<String> dets = Arrays.asList("A", "An", "The", "There");
		properTerms = properTerms.stream()
				.distinct().map(phrase -> removeEndPunct(phrase))
				.filter(phrase -> !isInt(phrase)).filter(phrase -> !dets.contains(phrase)).filter(phrase -> isValidWiki(phrase))
				.collect(Collectors.toList());		
		properTerms = removeSubsets(properTerms);
		System.out.println("Proper terms:" + properTerms.toString());
		return (ArrayList<String>) properTerms;
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
	    	compoundPhrase = compoundWordList.get(0).originalText().toLowerCase();
	    	for(int i = 1; i < compoundWordList.size(); i++) {
	    		compoundPhrase += " ";
	    		compoundPhrase += compoundWordList.get(i).originalText().toLowerCase();
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
		    String topicPhrase  = topicWords.get(0).toString().toLowerCase();	    
		    for(int j = 1; j < topicWords.size(); j++) {
		    	boolean endsInPossessive = (j == topicWords.size()-1) && topicWords.get(j).toString().equals("\'s");
		    	if(!endsInPossessive) {
			    	topicPhrase += " ";
			    	topicPhrase += topicWords.get(j).toString().toLowerCase();
		    	}
		    }
		    
		    
		    if(topicPhrase.startsWith("the ") || topicPhrase.startsWith("a ") || topicPhrase.startsWith("an ")) { 
		    	boolean proper = Character.isUpperCase(topicWords.get(0).toString().charAt(0)) && !claim.toLowerCase().startsWith(topicPhrase);
		    	boolean notProper = Character.isLowerCase(topicWords.get(0).toString().charAt(0)) && !claim.toLowerCase().startsWith(topicPhrase);
		    	String nonDetPhrase;
		    	if(topicPhrase.startsWith("the ")) {
		    		nonDetPhrase = topicPhrase.substring(4);
		    	}else if(topicPhrase.startsWith("an ")){
		    		nonDetPhrase = topicPhrase.substring(3);
		    	}else {
		    		nonDetPhrase = topicPhrase.substring(2);
		    	}
		    	
		    	if(proper) {
		    		if(isValidWiki(topicPhrase)) {
				    	nounPhrases.add(topicPhrase);
				    }
		    	} else if(notProper) {
		    		if(isValidWiki(nonDetPhrase)) {
			    		nounPhrases.add(nonDetPhrase);
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
		    else if(isValidWiki(topicPhrase)) {
		    	nounPhrases.add(topicPhrase);
		    }
		    i++;
	    }
	    
	    return nounPhrases;
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
				if(str.contains(subStr) && !str.equals(subStr)) {
					unique = false;
				}
		    }
			if(unique) {
				filtered.add(subStr);
			}
		}
		return filtered;
	}
	
	private static boolean isValidWiki(String title) {
		boolean valid = false;
		String wikiKey = title.toLowerCase().replaceAll(" ", "_").replace("(", "-lrb-").replace(")", "-rrb-").replace("]", "-rsb-").replace("[", "-lsb-");
		if(wikiMap.containsKey(wikiKey) || disambiguationMap.containsKey(wikiKey)) {
			valid = true;
	    }
		return valid;
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
		List<Tree> leaves = constituencyTree.getLeaves();
	    int wordNum = 0;

	    ArrayList<String> wordList = new ArrayList<String>();
	    String[] nounsTags = {"NN", "NNS", "NNP", "NNPS", "NP"};//, "CD", "JJ"}; 
	    while(wordNum < leaves.size()) {
	    	Tree word = leaves.get(wordNum);
	    	Tree type = word.parent(constituencyTree);
	    	String wordStr = word.toString().toLowerCase();
	    	if(Arrays.asList(nounsTags).contains(type.label().toString()) && !wikiTitle.contains(wordStr)) {
	    		wordList.add(wordStr);
	    	} 
	    	wordNum++;
	    }
    
	    return wordList;
	}

	private static ArrayList<String[]> getNounsAndNamedEntities(String wikiTitle,  Tree constituencyTree, List<String[]> namedEntities){
		ArrayList<String> nouns = getNouns(wikiTitle, constituencyTree);
	    List<String[]> filteredNamedEntities = namedEntities.stream().filter(arr -> !wikiTitle.contains(arr[0]))
				.collect(Collectors.toList());
	    System.out.println("claim named entities: ");
	    for(String[] ne : filteredNamedEntities) {
	    	System.out.print(ne[0] + "(" + ne[1] + "), ");
	    }
	    System.out.println();
			
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

	private static boolean containsNamedEntities(String evidenceSentence, String claim, ArrayList<String[]> claimNE, String wikiTitle, StanfordCoreNLP pipeline, String root) {
		boolean containsEntities = true;
		//unnecessary if statements are included here for the benefit of a descriptive text log
		if(evidenceSentence.isEmpty()) {
			containsEntities = false;
		}
		else if(claimNE.isEmpty()) {
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("contains no entities.");
			containsEntities = false;
		}
		
		String[] entityToBeSwapped = null;
		for(String[] ne: claimNE) {
			if(!evidenceSentence.toLowerCase().contains(ne[0]) && entityToBeSwapped == null) {
				entityToBeSwapped = ne.clone();
			}
			else if(!evidenceSentence.toLowerCase().contains(ne[0]) && containsEntities) {
				System.out.println("sentence: " + evidenceSentence);
				System.out.println("Missing both: \"" + ne[0] + "\" and \"" + entityToBeSwapped[0] + "\"");
				containsEntities = false;
			}
			
		}
		if(entityToBeSwapped == null) {
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("contains all entites");
		}
		if(containsEntities && entityToBeSwapped != null && entityToBeSwapped[1] == "NOUN") {
			System.out.println("sentence: " + evidenceSentence);
			if(isNounComplement(claim) && isNounComplement(evidenceSentence)) {
				System.out.println("The only remaining entity is a generic noun, and it's a 'is a' sentence.");
			} else {
				System.out.println("The only remaining entity is a generic noun, which can't be reliably replaced.");
				containsEntities = false;
			}
		}
		boolean missingRootVerb = false;
		if(isVerb(root, pipeline) && !evidenceSentence.contains(root)) {
			missingRootVerb = true;
		}
		if(containsEntities && entityToBeSwapped != null && entityToBeSwapped[1] != "NOUN" && missingRootVerb) {
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("The entity cannot be swapped because the root word is a verb which is missing.");
			containsEntities = false;
		}
		if(containsEntities && entityToBeSwapped != null && entityToBeSwapped[1] != "NOUN" && !missingRootVerb) {
			List<String[]> evidenceEntities = getNamedEntities(evidenceSentence, pipeline).stream().filter(arr -> !wikiTitle.contains(arr[0]))
					.collect(Collectors.toList());
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("entity " + entityToBeSwapped[0] + "(" + entityToBeSwapped[1] + ") is missing. Attempting to find replacement");
			System.out.print("sentence named entities: ");
		    for(String[] ne : evidenceEntities) {
		    	System.out.print(ne[0] + "(" + ne[1] + "), ");
		    }
		    System.out.println();
			containsEntities = false;
			for(String[] ne: evidenceEntities) {
				if(!containsEntities && ne[1].equals(entityToBeSwapped[1])) {
					containsEntities = true;
					System.out.println("Swap successful: " + ne[0] + " for " + entityToBeSwapped[0]);
				}
			}
		}
		
		return containsEntities;
	}
	
	private static boolean isNounComplement(String sentence) {
		String[] NCTags = {" is a ", " is an " , " was a " , " was an "};
		boolean isNC = false;
		for(String tag : NCTags) {
			if(sentence.contains(tag)) {
				isNC = true;
			}
		}
		return isNC;
	}
	
	private static boolean containsCorrelatedWord(String sentence, String root, String wikiTitle) {
		ArrayList<String> words = getWords(wikiTitle, sentence.toLowerCase());
		boolean correlated = false;
		if(correlationMap.containsKey(root)) {
			Map<String, Float> correlations = correlationMap.get(root);	
			for(String word: words) {
				if(correlations.containsKey(word)) {
					System.out.println("Sentence: " + sentence);
					System.out.println("Added via correlation of root " + root + " to word " + word);
					correlated = true;
				}
			}
		}
		return correlated;
	}
	
	private static ArrayList<String> getWords(String wikiTitle, String claim) {
		String[] words = claim.split(" |\\t");
		
	    ArrayList<String> wordList = new ArrayList<String>();
	    String[] skipWords = {"a", "an", "the", "at", "by", "down", "for", "from", "in", "into", "like", "near", "of", "off", "on", "onto", "onto", "over", 
				"past", "to", "upon", "with", "and", "&", "as", "but", "for", "if", "nor", "once", "or", "so", "than", "that", "till", "when", "yet", "'s", "'",
				"be", "is", "am", "are", "was", "were", "been", "being", "has", "have", "had", "having", "do", "does", "did",
				"he", "his", "him", "she", "her", "hers", "it", "its", "they", "theirs",
				".","!","?",",",";",":", "-rrb-", "-lrb-", "-rsb-", "-lsb-", "", "0", "``", "''", "--", "-"};
	    
	    for(int i = 0; i < words.length; i++) {
	    	String word = words[i];
	    	if(!Arrays.asList(skipWords).contains(word) && !wikiTitle.contains(word)) {
	    		wordList.add(word);
	    	} 
	    }

	    wordList = (ArrayList<String>) wordList.stream().distinct().collect(Collectors.toList());
	    return wordList;
	}
	
	private static boolean containsValidRoot(String sentence, String root, StanfordCoreNLP pipeline) {
		String[] isWords = {"is", "was", "be", "are", "were"};
		String rootAlone = " " + root + " ";
		boolean validRoot = false;
		if(isVerb(root, pipeline) && !Arrays.asList(isWords).contains(root) && sentence.toLowerCase().contains(rootAlone)) {
			System.out.println("Sentence: " + sentence);
			System.out.println("Added via presence of root " + root);
			validRoot = true;
		}
		return validRoot;
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
			String urlTitle = topic.replace(' ', '_').toLowerCase();
			Map<String, String> wikiDoc = new HashMap<String, String>();
			boolean emptyDisam = false;
			if (!topic.isEmpty() && wikiMap.containsKey(urlTitle)){
				Map<String, Object> fileInfo = wikiMap.get(urlTitle);
				String wikiListName = wikiDirName + "\\" + fileInfo.get("fileName");
				try {
					BufferedReader reader = new BufferedReader(new FileReader(wikiListName)); 
					Long byteOffset = (Long) fileInfo.get("offset");
					reader.skip(byteOffset);
				    String wikiEntry = reader.readLine();
				    JSONObject wikiJson = new JSONObject(wikiEntry);
				    if (wikiJson.getString("text").toLowerCase().contains((topic+" may refer to : "))) {
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
						if(disambiguations.contains(wiki.get("id").toLowerCase())) {
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
				String wikiListName = wikiDirName + "\\" + fileInfo.get("fileName");
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
	
	private static ArrayList<String> getBackupDocs(ArrayList<String> possibleTopics, ArrayList<Map<String, String>> existingDocs) {
		ArrayList<String> backupDocs = new ArrayList<String>();
		for(String topic: possibleTopics) {
			String urlTitle = topic.replace(' ', '_').toLowerCase();
			if (!topic.isEmpty() && disambiguationMap.containsKey(urlTitle)){
				backupDocs.addAll(disambiguationMap.get(urlTitle));
				for(Map<String, String> wiki: existingDocs) {
					if(backupDocs.contains(wiki.get("id").toLowerCase())) {
						backupDocs.remove(wiki.get("id").toLowerCase());
					}
				}
			}
		
		}
		System.out.println("final topics: " + backupDocs.toString());
		return backupDocs;
	}
	
	private static ArrayList<Map<String, String>> findDisambiguationChildren(JSONObject disambiguation){
		ArrayList<Map<String, String>> disambiguationChildren = new ArrayList<Map<String, String>>();
		try {
			String lines = disambiguation.getString("lines");
			String[] entries = lines.split("[\\n[\\d+]\\t]+");
			ArrayList<String> topics = new ArrayList<String>();
			for(int i = 1; i < entries.length; i++) {
				String normal = entries[i].replaceAll(" , ", ", ").replaceAll(" : ", ": ").replaceAll(" ; ", "; ").replaceAll(" \'", "\'").replaceAll(" -- ", "�").replaceAll("-LRB- ", "-LRB-").replaceAll(" -RRB-", "-RRB-");
				int end = normal.indexOf(',');
				while(end > 0) {
					String childPhrase = normal.substring(0, end);
					if(!childPhrase.replace(' ', '_').equalsIgnoreCase(disambiguation.getString("id"))) {
						topics.add(childPhrase);
					}
					end = normal.indexOf(',', end + 1);
				}
				topics.add(normal);
			}
			disambiguationChildren = getDocsFromTopics(topics);
		} catch (JSONException e) {
			e.printStackTrace();
		}
		
		return disambiguationChildren;
	}
	
	public static void getWikiMap(String wikiDirName) {
		wikiMap = new HashMap<String, Map <String, Object>>();
		disambiguationMap = new HashMap<String, ArrayList<String>>();
		
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
						String wikiEntry = Normalizer.normalize(s.nextLine(), Normalizer.Form.NFD);
					    JSONObject wikiJson = new JSONObject(wikiEntry);
					    String id = wikiJson.getString("id").toLowerCase();
					    if(!id.isEmpty()) {
					    	String fileName = wikiEntryList.getName();
					    	Map<String, Object> docLocation = new HashMap<String, Object>();
					    	docLocation.put("fileName", fileName);
					    	docLocation.put("offset", byteOffset);
							wikiMap.put(id, docLocation);
							
							int paren = id.indexOf("-lrb-");
							if(paren > 0 && !id.contains("disambiguation")) {
								String base = id.substring(0, paren-1).toLowerCase();
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

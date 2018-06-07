import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

import org.json.*;

//import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
//import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
//import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
//import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
//import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
//import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;
//import edu.stanford.nlp.util.CoreMap;


public class DocFinder {
	static String claimsFileName = "shared_task_dev_public.jsonl";
	static String resultsFileName = "results.jsonl";
	static String wikiDirName = "wiki-dump";
	static int numClaimsToTest = 100;
	
	static Map<String, Map<String, Object>> wikiMap;
	static Map<String, ArrayList<String>> disambiguationMap;
	
	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    System.out.println("Beginning document processing. Time: "+dtf.format(LocalDateTime.now()));	    
		
		String statement = "";
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma, ner");
	    props.setProperty("coref.algorithm", "neural");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(LocalDateTime.now()));
		try {
			Scanner claimReader = new Scanner(new FileReader(claimsFileName));
			File oldFile = new File(resultsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFileName, true));
			getWikiMap(wikiDirName);
			System.out.println("wikiMap compiled. Time: "+dtf.format(LocalDateTime.now()));
			int claimCount =0;
			while(claimReader.hasNext() && claimCount < numClaimsToTest) {
				claimCount++;		
			    statement = claimReader.nextLine();
			    JSONObject claimJson = new JSONObject(statement);
			    String claim = claimJson.getString("claim");
			    
			    System.out.println("Claim "+ claimCount + ":  \""+claim+"\" Time: "+dtf.format(LocalDateTime.now()));
			    //Find potentially relevant documents
			    ArrayList<String> claimTopics = getClaimTopics(pipeline, claim); 
			    //System.out.println("Beginning document retrieval. Time: "+dtf.format(LocalDateTime.now()));
			    ArrayList<Map<String, String>> possibleWikiDocs = getDocsFromTopics(claimTopics);
			    ArrayList<String> backupDocs = getBackupDocs(claimTopics, possibleWikiDocs);
			    
			    //Get potentially relevant sentences
			    //System.out.println("Beginning sentence retrieval. Time: "+dtf.format(LocalDateTime.now()));
			    Map<String, Object> wikiInfo = new HashMap<String, Object>();
			    for(Map<String, String> wikiDoc: possibleWikiDocs) {
			    	Map<Integer, String> matchingSentences = new HashMap<Integer, String>();
				    ArrayList<Integer> matchingSentenceNums = getTextMatches(claim, wikiDoc.get("text"));
				    //matchingSentenceNums.addAll(getNamedEntityMatches(claim, wikiDoc.get("text"), pipeline));
				    matchingSentenceNums = (ArrayList<Integer>) matchingSentenceNums.stream().distinct().collect(Collectors.toList());
				    String[] textSentences = wikiDoc.get("text").split(" [.] ");
				    for(int i = 0; i < textSentences.length; i++) {
				    	if(matchingSentenceNums.contains(i)) {
				    		matchingSentences.put(i, textSentences[i]);
				    	}
				    }
				    wikiInfo.put(wikiDoc.get("id"), matchingSentences);
			    }
			    //TODO: add key word sentence finder (possibly remove name entity)
			    
			    //TODO: repeat process for backup wikis if no relevant sentences found
			    
			    //determine if sentences are verify, refute, or not enough info
			    //TODO: implement classifier
			    
			    //System.out.println("Printing to JSON. Time: "+dtf.format(LocalDateTime.now()));
			    //convert and print to JSON
			    JSONObject result = convertToJSON(claimJson.getInt("id"), claim, wikiInfo, backupDocs);
			    writer.append(result.toString());
			    writer.append("\n");
			    
			    //System.out.println("Claim "+ claimCount + " complete. Time: "+dtf.format(LocalDateTime.now())+ "\n");
			}
			claimReader.close();
			writer.close();
		} catch (FileNotFoundException e) {
			System.out.println("Could not open file  "+claimsFileName);
			e.printStackTrace();
		} catch (JSONException e) {
			System.out.println("There was a problem with the json from " + statement);
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("There was an IO Exception during statement " + statement);
			e.printStackTrace();
		}
		
		System.out.println("Document processing finished. Time: "+dtf.format(LocalDateTime.now()));
	}
	
	
	private static ArrayList<String> getClaimTopics(StanfordCoreNLP pipeline, String claimSentence) {
		GrammaticalRelation[] subjectObjectRelations = {UniversalEnglishGrammaticalRelations.SUBJECT, UniversalEnglishGrammaticalRelations.CLAUSAL_PASSIVE_SUBJECT, 
	    		UniversalEnglishGrammaticalRelations.CLAUSAL_SUBJECT, //UniversalEnglishGrammaticalRelations.CONTROLLING_CLAUSAL_PASSIVE_SUBJECT,
	    		UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT, UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT,
	    		UniversalEnglishGrammaticalRelations.DIRECT_OBJECT,UniversalEnglishGrammaticalRelations.OBJECT};
		GrammaticalRelation[] dependencyRelations= {UniversalEnglishGrammaticalRelations.APPOSITIONAL_MODIFIER, UniversalEnglishGrammaticalRelations.CLAUSAL_COMPLEMENT, 
				UniversalEnglishGrammaticalRelations.XCLAUSAL_COMPLEMENT, UniversalEnglishGrammaticalRelations.POSSESSION_MODIFIER, 
				UniversalEnglishGrammaticalRelations.NAME_MODIFIER, UniversalEnglishGrammaticalRelations.NOMINAL_MODIFIER, UniversalEnglishGrammaticalRelations.CASE_MARKER,
				UniversalEnglishGrammaticalRelations.getNmod("of")};		
	    
		CoreDocument document = new CoreDocument(claimSentence);
	    pipeline.annotate(document);
	    CoreSentence claimSen = document.sentences().get(0);
	    SemanticGraph dependencyGraph = claimSen.dependencyParse();
	    //System.out.println("Dependency graph: "+dependencyGraph);
	    
	    ArrayList<IndexedWord> topicList = new ArrayList<IndexedWord>();
	    Set<IndexedWord> topicWords = new HashSet<IndexedWord>();
	    topicWords.add(dependencyGraph.getFirstRoot());
		for(int dependencyRelation = 0; dependencyRelation < dependencyRelations.length; dependencyRelation++) {
			Set<IndexedWord> topicDependency = new HashSet<IndexedWord>();
			for(IndexedWord topicWord: topicWords) {
				topicDependency.addAll(getDescendentsWithReln(dependencyGraph, topicWord, dependencyRelations[dependencyRelation], 2));
			}	
    		if(!topicDependency.isEmpty()) {
    			//System.out.println(topicDependency.toString() + " added via " + dependencyRelations[dependencyRelation].toString());
    			topicList.addAll(topicDependency);
    		}
	    }
		
	    for(int relation = 0; relation < subjectObjectRelations.length; relation++) {
	    	topicWords = getDescendentsWithReln(dependencyGraph, dependencyGraph.getFirstRoot(), subjectObjectRelations[relation], 2);
	    	if(!topicWords.isEmpty()) {
	    		//System.out.println(topicWords.toString() + " added via " + subjectObjectRelations[relation].toString());
    			topicList.addAll(topicWords);
    			for(int dependencyRelation = 0; dependencyRelation < dependencyRelations.length; dependencyRelation++) {
    				Set<IndexedWord> topicDependency = new HashSet<IndexedWord>();
    				for(IndexedWord topicWord: topicWords) {
    					topicDependency.addAll(getDescendentsWithReln(dependencyGraph, topicWord, dependencyRelations[dependencyRelation], 2));
    				}
    				if(!topicDependency.isEmpty()) {
    	    			//System.out.println(topicDependency.toString() + " added via " + dependencyRelations[dependencyRelation].toString());
    	    			topicList.addAll(topicDependency);
    	    		}
    		    }
    		}
	    }

	    Tree constituencyTree = claimSen.constituencyParse();
	    //System.out.println("constituencyTree: " +constituencyTree);
	    
	    //DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    //System.out.println("topics gathered. List size: "+ topicList.size()+". Time: "+dtf.format(LocalDateTime.now()));
	    ArrayList<String> topicPhrases = new ArrayList<String>();
	    for(int h = 0; h < topicList.size(); h++) {
	    	IndexedWord topicWord = topicList.get(h);
	    	//System.out.println("topic word: "+ topicWord.originalText()+". Time: "+dtf.format(LocalDateTime.now()));
	    	String compoundPhrase = getCompoundPhrase(topicWord, dependencyGraph);
	    	if(compoundPhrase != null) {
	    		//System.out.println("compound Phrase: " +compoundPhrase);
	    		topicPhrases.add(compoundPhrase);
	    	}
	    	//System.out.println("Compound phrase done. Time: "+dtf.format(LocalDateTime.now()));
	    	
	    	ArrayList<String> nounPhrases = getNounPhrases(claimSentence, topicWord, constituencyTree);

		    topicPhrases.addAll(nounPhrases);
		    //System.out.println("Noun phrases done. Time: "+dtf.format(LocalDateTime.now()));
		    
		  //TODO: add named entities with better version, and get rid of redundancies
	    }
	    
	    
	    //get rid of duplicates
	    topicPhrases = (ArrayList<String>) topicPhrases.stream().distinct().collect(Collectors.toList());
		return topicPhrases;
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
	    	compoundPhrase = "";
	    	compoundPhrase += compoundWordList.get(0).originalText().toLowerCase();
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
	    String[] nounsAndPhrases = {"NN", "NNS", "NNP", "NNPS", "NP"};//, "VP", "CONJP"};
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
		    //System.out.println("Attempted noun phrase: " + topicPhrase);
		   
		    
		    
		    if(topicPhrase.startsWith("the ")) {
		    	boolean proper = topicWords.get(0).toString().equals("The") && !claim.toLowerCase().startsWith(topicPhrase);
		    	boolean notProper = topicWords.get(0).toString().equals("the") && !claim.toLowerCase().startsWith(topicPhrase);
		    	String nonDetPhrase = topicPhrase.substring(4);
		    	if(proper) {
		    		if(wikiMap.containsKey(topicPhrase.replaceAll(" ", "_")) || disambiguationMap.containsKey(topicPhrase.replaceAll(" ", "_"))) {
				    	nounPhrases.add(topicPhrase);
				    }
		    	}
		    	else if(notProper) {
		    		if(wikiMap.containsKey(nonDetPhrase.replaceAll(" ", "_")) || disambiguationMap.containsKey(nonDetPhrase.replaceAll(" ", "_"))) {
			    		nounPhrases.add(nonDetPhrase);
			    	}
		    	}
		    	else {
		    		if(wikiMap.containsKey(topicPhrase.replaceAll(" ", "_")) || disambiguationMap.containsKey(topicPhrase.replaceAll(" ", "_"))) {
				    	nounPhrases.add(topicPhrase);
				    }
		    		if(wikiMap.containsKey(nonDetPhrase.replaceAll(" ", "_")) || disambiguationMap.containsKey(nonDetPhrase.replaceAll(" ", "_"))) {
			    		nounPhrases.add(nonDetPhrase);
			    	}
		    	}
		    }
		    else if(wikiMap.containsKey(topicPhrase.replaceAll(" ", "_")) || disambiguationMap.containsKey(topicPhrase.replaceAll(" ", "_"))) {
		    	nounPhrases.add(topicPhrase);
		    }
		    i++;
	    }
	    
	    return nounPhrases;
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
	
	private static ArrayList<Integer> getTextMatches(String sentence, String text){
		String phrase = sentence.substring(0, sentence.length()-1);
		ArrayList<String> matchingSentences = new ArrayList<String>();
		ArrayList<Integer> matchingSentenceNums = new ArrayList<Integer>();
		String[] textSentences = text.split(" \\.|\\!|\\? ");
		for(int i = 0; i < textSentences.length; i++) {
			String normal = textSentences[i].replaceAll(" , ", ", ").replaceAll(" : ", ": ").replaceAll(" ; ", "; ").replaceAll(" '", "'").replaceAll(" -- ", "–").replaceAll(" -LRB- ", " (").replaceAll("-RRB- ", ") ").replaceAll("-LSB- ", " [").replaceAll("-RSB- ", "] ");
			String negateSentence = normal.replaceAll("not ", "").replaceAll("n\'t", "").replaceAll("never ", "");
			if(normal.contains(phrase)) {
				matchingSentences.add(textSentences[i]);
				matchingSentenceNums.add(i);
				System.out.println("full phrase match found");
			} else if(negateSentence.contains(phrase)) {
				matchingSentences.add(textSentences[i]);
				matchingSentenceNums.add(i);
				System.out.println("negation match found");
			}
		}
		
		return matchingSentenceNums;
	}
	
	//possibly remove subject of article from included named entities?
//	private static ArrayList<Integer> getNamedEntityMatches(String sentence, String text, StanfordCoreNLP pipeline){
//		ArrayList<String> namedEntities = getNamedEntities(sentence, pipeline);
//		ArrayList<String> matchingSentences = new ArrayList<String>();
//		ArrayList<Integer> matchingSentenceNums = new ArrayList<Integer>();
//		String[] textSentences = text.split(" [.] ");
//		for(int i = 0; i < textSentences.length; i++) {
//			//normalize sentence for matching
//			String normalSent = textSentences[i].replaceAll(" , ", ", ").replaceAll(" : ", ": ").replaceAll(" ; ", "; ").replaceAll(" '", "'").replaceAll(" -- ", "–").replaceAll(" -LRB- ", " (").replaceAll("-RRB- ", ") ").replaceAll("-LSB- ", " [").replaceAll("-RSB- ", "] ");
//			boolean includesAll = true;
//			for(int j = 0; j < namedEntities.size(); j++) {
//				if(!normalSent.contains(namedEntities.get(j))) {
//					includesAll = false;
//				}
//			}
//			if(includesAll) {
//				matchingSentences.add(textSentences[i]);
//				matchingSentenceNums.add(i);
//			}
//		}
//		
//		return matchingSentenceNums;
//	}
	
//	private static ArrayList<String> getNamedEntities(String sentence, StanfordCoreNLP pipeline){
//		ArrayList<CoreLabel> tokens = getSentenceTokens(sentence, pipeline);
//		ArrayList<String> namedEntities = new ArrayList<String>();
//		String namedEntity = "";
//		boolean neActive = false;
//		for(int i = 0; i < tokens.size(); i++) {
//			String ne = tokens.get(i).get(NamedEntityTagAnnotation.class);
//			if(!neActive && !ne.equals("O")) {
//				neActive = true;
//				namedEntity += tokens.get(i).get(TextAnnotation.class);
//			}
//			else if(neActive && !ne.equals("O")) {
//				namedEntity += " " + tokens.get(i).get(TextAnnotation.class);
//			}
//			else if(neActive && ne.equals("O")) {
//				neActive = false;
//				namedEntities.add(namedEntity);
//				namedEntity = "";
//			}
//		}
//		if(neActive) {
//			namedEntities.add(namedEntity);
//		}
//		return namedEntities;
//	}
	
//	private static ArrayList<CoreLabel> getSentenceTokens(String sentence, StanfordCoreNLP pipeline){
//		Annotation document = new Annotation(sentence);
//	    pipeline.annotate(document);
//	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
//	    CoreMap tokenSen = sentences.get(0);
//	    
//	    ArrayList<CoreLabel> tokens = new ArrayList<CoreLabel>();
//	    for (CoreLabel token: tokenSen.get(TokensAnnotation.class)) {
//	        tokens.add(token);
//	      }
//	    return tokens;
//	}
	
	private static ArrayList<Map<String, String>> getDocsFromTopics(ArrayList<String> possibleTopics) {
		//DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");
		ArrayList<Map<String, String>> wikiDocs = new ArrayList<Map<String, String>>();
		//System.out.println("Number of topics for docs: "+possibleTopics.size());
		for(String topic: possibleTopics) {
			String urlTitle = topic.replace(' ', '_').toLowerCase();
			Map<String, String> wikiDoc = new HashMap<String, String>();
			boolean emptyDisam = false;
			if (!topic.isEmpty() && wikiMap.containsKey(urlTitle)){
				//System.out.println("valid topic: " + topic + " Time: " + dtf.format(LocalDateTime.now()));
				Map<String, Object> fileInfo = wikiMap.get(urlTitle);
				String wikiListName = wikiDirName + "\\" + fileInfo.get("fileName");
				//System.out.println("Key found. Time: "+dtf.format(LocalDateTime.now()));
				try {
					BufferedReader reader = new BufferedReader(new FileReader(wikiListName)); 
					Long byteOffset = (Long) fileInfo.get("offset");
					reader.skip(byteOffset);
				    String wikiEntry = reader.readLine();
				    JSONObject wikiJson = new JSONObject(wikiEntry);
				    //System.out.println("JSON retrieved. Time: " + dtf.format(LocalDateTime.now()));
				    if (wikiJson.getString("text").toLowerCase().contains((topic+" may refer to : "))) {
				    	//System.out.println("disambiguation page: " + topic);
			    		ArrayList<Map<String, String>> disambiguationChildren = findDisambiguationChildren(wikiJson);
			    		wikiDocs.addAll(disambiguationChildren);
			    		if(disambiguationChildren.isEmpty()) {
			    			//System.out.println("disambiguation page empty");
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
				//System.out.println("disambiguation found: " + urlTitle+". Time: "+dtf.format(LocalDateTime.now()));
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
			//System.out.println("Topic " + topic + " done. Time: " + dtf.format(LocalDateTime.now()));
		}
		return wikiDocs;
	}
	
	private static ArrayList<String> getBackupDocs(ArrayList<String> possibleTopics, ArrayList<Map<String, String>> existingDocs) {
		ArrayList<String> backupDocs = new ArrayList<String>();
		for(String topic: possibleTopics) {
			String urlTitle = topic.replace(' ', '_').toLowerCase();
			
			if (!topic.isEmpty() && disambiguationMap.containsKey(urlTitle)){
				backupDocs.addAll(disambiguationMap.get(urlTitle));
				for(Map<String, String> wiki: existingDocs) {
					if(backupDocs.contains(wiki.get("id").toLowerCase())) {
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
				String normal = entries[i].replaceAll(" , ", ", ").replaceAll(" : ", ": ").replaceAll(" ; ", "; ").replaceAll(" \'", "\'").replaceAll(" -- ", "–").replaceAll("-LRB- ", "-LRB-").replaceAll(" -RRB-", "-RRB-");
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
	
	
	
	private static JSONObject convertToJSON(int claimID, String claim, Map<String, Object> wikiInfo, ArrayList<String> backupDocs) {
		Map<String, Object> jsonMap = new LinkedHashMap<String, Object>();
		jsonMap.put("id", claimID);
		jsonMap.put("claim", claim);
		jsonMap.put("wiki info", wikiInfo);
		jsonMap.put("backup wikis", backupDocs.toArray(new String[backupDocs.size()]));
				
		return new JSONObject(jsonMap);
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
							String wikiEntry = s.nextLine();
						    JSONObject wikiJson = new JSONObject(wikiEntry);
						    String id = wikiJson.getString("id").toLowerCase();
						    if(!id.isEmpty()) {
						    	String fileName = wikiEntryList.getName();
						    	Map<String, Object> docLocation = new HashMap<String, Object>();
						    	docLocation.put("fileName", fileName);
						    	docLocation.put("offset", byteOffset);
								wikiMap.put(id, docLocation);
								
								int paren = id.indexOf("-lrb-");
								if(paren > 0) {
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

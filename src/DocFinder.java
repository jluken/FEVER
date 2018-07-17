import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.Normalizer;
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

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;


public class DocFinder {
	static String claimsFileName = "shared_task_dev_public.jsonl";
	static String resultsFileName = "results.jsonl";
	static String wikiDirName = "wiki-dump";
	static int numClaimsToTest = 200;
	
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
			    statement = Normalizer.normalize(claimReader.nextLine(), Normalizer.Form.NFD);
			    JSONObject claimJson = new JSONObject(statement);
			    String claim = claimJson.getString("claim");
			    
			    System.out.println("Claim "+ claimCount + ":  \""+claim+"\" Time: "+dtf.format(LocalDateTime.now()));
			    ArrayList<String> claimTopics = getProperTerms(claim);
			    if(claimTopics.isEmpty()) {
			    	claimTopics = getAllTopics(pipeline, claim);
			    }

			    ArrayList<Map<String, String>> possibleWikiDocs = getDocsFromTopics(claimTopics);
			    ArrayList<String> backupDocs = getBackupDocs(claimTopics, possibleWikiDocs);

			    Map<String, Object> wikiInfo = new HashMap<String, Object>();
			    for(Map<String, String> wikiDoc: possibleWikiDocs) {
				    wikiInfo.put(wikiDoc.get("id"), new HashMap<Integer, String>());
			    }
			    JSONObject result = convertToJSON(claimJson.getInt("id"), claim, wikiInfo, backupDocs);
			    writer.append(result.toString());
			    writer.append("\n");
			    
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
	
	private static ArrayList<String> getProperTerms(String sentence){
		String[] lowerWords = {"a", "an", "the", "at", "by", "down", "for", "from", "in", "into", "like", "near", "of", "off", "on", "onto", "onto", "over", 
				"past", "to", "upon", "with", "and", "&", "as", "but", "for", "if", "nor", "once", "or", "so", "than", "that", "till", "when", "yet"};
		ArrayList<String> properTerms = new ArrayList<String>();
		List<String> possibleProperTerms = new ArrayList<String>();
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
				possibleProperTerms.add(properPhrase);
				properPhrase += " " + word;
			}
			else if(!properPhrase.isEmpty() && (word.startsWith("(") || word.startsWith("["))) {
				possibleProperTerms.add(properPhrase);
				paren = true;
				properPhrase += " " + word;
			}
			else if(!properPhrase.isEmpty() && (word.endsWith(")") || word.endsWith("]"))) {
				paren = false;
				properPhrase += word;
				possibleProperTerms.add(properPhrase);
				properPhrase = "";
			}
			else if(!properPhrase.isEmpty()){
				possibleProperTerms.add(properPhrase);
				properPhrase = "";
			}
		}
		if(!properPhrase.isEmpty()) {
			possibleProperTerms.add(properPhrase);
		}
		System.out.println("Possible proper terms:" + possibleProperTerms.toString());
		List<String> dets = Arrays.asList("A", "An", "The", "There");
		possibleProperTerms = possibleProperTerms.stream()
				.distinct().map(phrase -> removeEndPunct(phrase))
				.filter(phrase -> !isInt(phrase)).filter(phrase -> !dets.contains(phrase)).filter(phrase -> isValidWiki(phrase))
				.collect(Collectors.toList());		
		possibleProperTerms = removeSubsets(possibleProperTerms);
		System.out.println("Proper terms:" + properTerms.toString());
		return properTerms;
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
	
	private static boolean isValidWiki(String title) {
		boolean valid = false;
		String wikiKey = title.toLowerCase().replaceAll(" ", "_").replace("(", "-lrb-").replace(")", "-rrb-").replace("]", "-rsb-").replace("[", "-lsb-");
		if(wikiMap.containsKey(wikiKey) || disambiguationMap.containsKey(wikiKey)) {
			valid = true;
	    }
		return valid;
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
	
	private static boolean isInt(String str) {
		boolean isInt = true;
		try {
			Integer.parseInt(str);
	    } catch (NumberFormatException e) {
	        isInt = false;
	    }
		return isInt;
	}
	
	
	private static ArrayList<String> getAllTopics(StanfordCoreNLP pipeline, String claimSentence) {
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

	    ArrayList<String> topicPhrases = new ArrayList<String>();
	    for(int h = 0; h < topicList.size(); h++) {
	    	IndexedWord topicWord = topicList.get(h);
	    	String compoundPhrase = getCompoundPhrase(topicWord, dependencyGraph);
	    	if(compoundPhrase != null) {
	    		//System.out.println("compound Phrase: " +compoundPhrase);
	    		topicPhrases.add(compoundPhrase);
	    	}
	    	
	    	ArrayList<String> nounPhrases = getNounPhrases(claimSentence, topicWord, constituencyTree);

		    topicPhrases.addAll(nounPhrases);
		    //System.out.println("Noun phrases done. Time: "+dtf.format(LocalDateTime.now()));
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
		    
		    
		    if(topicPhrase.startsWith("the ") || topicPhrase.startsWith("a ")) { 
		    	boolean proper = Character.isUpperCase(topicWords.get(0).toString().charAt(0)) && !claim.toLowerCase().startsWith(topicPhrase);
		    	boolean notProper = Character.isLowerCase(topicWords.get(0).toString().charAt(0)) && !claim.toLowerCase().startsWith(topicPhrase);
		    	String nonDetPhrase;
		    	if(topicPhrase.startsWith("the ")) {
		    		nonDetPhrase = topicPhrase.substring(4);
		    	}else {
		    		nonDetPhrase = topicPhrase.substring(2);
		    	}
		    	
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

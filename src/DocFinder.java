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
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

import org.json.*;

import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;
import edu.stanford.nlp.util.CoreMap;


public class DocFinder {
	static String statementsFileName = "shared_task_dev_public.jsonl";
	static String resultsFileName = "results.jsonl";
	static String wikiDirName = "wiki-dump";
	static int numClaimsToTest = 100;
	
	
	
	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    LocalDateTime now = LocalDateTime.now();  
	    //System.out.println("Beginning document processing. Time: "+dtf.format(now));
	    
		GrammaticalRelation[] subjectRelations= {UniversalEnglishGrammaticalRelations.SUBJECT, UniversalEnglishGrammaticalRelations.CLAUSAL_PASSIVE_SUBJECT, 
	    		UniversalEnglishGrammaticalRelations.CLAUSAL_SUBJECT, UniversalEnglishGrammaticalRelations.CONTROLLING_CLAUSAL_PASSIVE_SUBJECT,
	    		UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT, UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT};
		GrammaticalRelation[] objectRelations= {UniversalEnglishGrammaticalRelations.DIRECT_OBJECT,UniversalEnglishGrammaticalRelations.OBJECT};
		
		String statement = "";
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma, ner");
	    props.setProperty("coref.algorithm", "neural");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    now = LocalDateTime.now();
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(now));
		try {
			Scanner statementReader = new Scanner(new FileReader(statementsFileName));
			File oldFile = new File(resultsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFileName, true));
			Map<String, Object[]> wikiMap = getWikiMap(wikiDirName);
			now = LocalDateTime.now();
			System.out.println("wikiMap compiled. Time: "+dtf.format(now));
			int statementCount =0;
			while(statementReader.hasNext() && statementCount < numClaimsToTest) {
				statementCount++;
				now = LocalDateTime.now();
				System.out.println("Beginning statement "+ statementCount + ". Time: "+dtf.format(now));				
			    statement = statementReader.nextLine();
			    JSONObject stateJson = new JSONObject(statement);
			    String claimID = stateJson.get("id").toString();
			    String claim = stateJson.getString("claim");
			    ArrayList<String> claimSubjects = getClaimTopic(claim, subjectRelations, pipeline, false);
			    Map<String, String> mainDoc = new HashMap<String, String>();
			    String claimSubject = "";
			    int i =0;
			    while(mainDoc.isEmpty() && i < claimSubjects.size()) {
			    	claimSubject = claimSubjects.get(i);
			    	mainDoc = getDocFromTitle(wikiMap, claimSubject);
			    	i++;
			    	//System.out.println("ping "+dtf.format(now));
			    }
			    //System.out.println("Main Test 1: "+dtf.format(now));
			    if (mainDoc.isEmpty()) {
			    	ArrayList<String> claimSubjectDependents = getClaimTopic(claim, subjectRelations, pipeline, true);
				    String claimSubjectDependent = "";
				    i =0;
				    while(mainDoc.isEmpty() && i < claimSubjectDependents.size()) {
				    	claimSubjectDependent = claimSubjectDependents.get(i);
				    	mainDoc = getDocFromTitle(wikiMap, claimSubjectDependent);
				    	i++;
				    }
			    }
			    //System.out.println("Main Test 2: "+dtf.format(now));
			    if (mainDoc.isEmpty()) {
			    	ArrayList<String> claimObjects = getClaimTopic(claim, objectRelations, pipeline, false);
				    String claimObject = "";
				    i =0;
				    while(mainDoc.isEmpty() && i < claimObjects.size()) {
				    	claimObject = claimObjects.get(i);
				    	mainDoc = getDocFromTitle(wikiMap, claimObject);
				    	i++;
				    }
			    } 
			    //System.out.println("Main Test 3: "+dtf.format(now));
			    if (mainDoc.isEmpty()) {
			    	ArrayList<String> claimObjectDependents = getClaimTopic(claim, objectRelations, pipeline, true);
				    String claimObjectDependent = "";
				    i =0;
				    while(mainDoc.isEmpty() && i < claimObjectDependents.size()) {
				    	claimObjectDependent = claimObjectDependents.get(i);
				    	mainDoc = getDocFromTitle(wikiMap, claimObjectDependent);
				    	i++;
				    }
			    }
			    //System.out.println("Main Test 4: "+dtf.format(now));
			    if (mainDoc.isEmpty()) {
			    	System.out.println("Unable to find Document to match claim.");
			    	mainDoc.put("id", "no subject found");
			    	mainDoc.put("text", "");
			    }
			    now = LocalDateTime.now();
			    //System.out.println("Document found. Time: "+dtf.format(now));
			    
			    ArrayList<String> matchingSentences = new ArrayList<String>();
			    ArrayList<Integer> matchingSentenceNums = getTextMatches(claim, mainDoc.get("text"));
			    matchingSentenceNums.addAll(getNamedEntityMatches(claim, mainDoc.get("text"), pipeline));
			    matchingSentenceNums = (ArrayList<Integer>) matchingSentenceNums.stream().distinct().collect(Collectors.toList());
			    String[] textSentences = mainDoc.get("text").split(" [.] ");
			    for(i = 0; i < textSentences.length; i++) {
			    	if(matchingSentenceNums.contains(i)) {
			    		matchingSentences.add(textSentences[i]);
			    	}
			    }
			    		
			    JSONObject result = convertToJSON(claimID, claim, mainDoc.get("id"), matchingSentenceNums, matchingSentences);
			    writer.append(result.toString());
			    writer.append("\n");
			    
			    now = LocalDateTime.now();  
			    System.out.println("Statement "+ statementCount + " complete. Time: "+dtf.format(now));
			}
			statementReader.close();
			writer.close();
		} catch (FileNotFoundException e) {
			System.out.println("Could not open file  "+statementsFileName);
			e.printStackTrace();
		} catch (JSONException e) {
			System.out.println("There was a problem with the json from " + statement);
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("There was an IO Exception during statement " + statement);
			e.printStackTrace();
		}
		
		
		
	}
	
	
	private static ArrayList<String> getClaimTopic(String claimSentence, GrammaticalRelation[] relations, StanfordCoreNLP pipeline, boolean dependents) {
		GrammaticalRelation[] dependencyRelations= {UniversalEnglishGrammaticalRelations.APPOSITIONAL_MODIFIER, UniversalEnglishGrammaticalRelations.CLAUSAL_COMPLEMENT, 
				UniversalEnglishGrammaticalRelations.XCLAUSAL_COMPLEMENT, UniversalEnglishGrammaticalRelations.POSSESSION_MODIFIER};

		
	    CoreDocument document = new CoreDocument(claimSentence);
	    pipeline.annotate(document);
	    CoreSentence claimSen = document.sentences().get(0);
	    SemanticGraph claimDependencyParse = claimSen.dependencyParse();
	    
	    IndexedWord topic = null;
	    ArrayList<IndexedWord> topics = new ArrayList<IndexedWord>();
	    
	    if(dependents) {
	    	int relation = 0;
	    	IndexedWord topicParent = null;
	    	while(topicParent == null && relation < relations.length) {
	    		
	    		topicParent = getDescendentWithReln(claimDependencyParse, claimDependencyParse.getFirstRoot(), relations[relation], 2);
	    		relation++;
	    	}
	    	if(topicParent != null) {
				int dependencyRelation = 0;
		    	while(dependencyRelation < dependencyRelations.length) {
		    		topic = getDescendentWithReln(claimDependencyParse, topicParent, dependencyRelations[dependencyRelation], 2);
		    		if(topic != null) {
		    			topics.add(topic);
		    		}
			    	dependencyRelation++;
			    }
	    	}
	    	
	    }
	    else {
	    	int relation = 0;
		    while(topics.isEmpty() && relation < relations.length) {
		    	topic = getDescendentWithReln(claimDependencyParse, claimDependencyParse.getFirstRoot(), relations[relation], 2);
		    	if(topic != null) {
	    			topics.add(topic);
	    		}
		    	relation++;
		    }
	    }
	    //System.out.println("claimSen: " + claimSen.toString());
	    Tree claimConstituencyParse = claimSen.constituencyParse();
	    if(topics.isEmpty()) {
	    	ArrayList<String> nullReturn = new ArrayList<String>();
	    	nullReturn.add("No subject detected in sentence");
	    	return nullReturn;
	    }
	    
	    ArrayList<String> topicPhrases = new ArrayList<String>();
	    for(int h = 0; h < topics.size(); h++) {
	    	topic = topics.get(h);
	    	//get compound children of topic as unique phrase
	    	Set<IndexedWord> compoundPhraseSet = claimDependencyParse.getChildrenWithRelns(topic, java.util.Collections.singleton(UniversalEnglishGrammaticalRelations.COMPOUND_MODIFIER));
	    	ArrayList<IndexedWord> compoundPhraseList = new ArrayList<IndexedWord>();
	    	compoundPhraseList.add(topic);
	    	for (IndexedWord compound : compoundPhraseSet) {
	            int phraseIndex = 0;
	            while(phraseIndex < compoundPhraseList.size() && compound.index() > compoundPhraseList.get(phraseIndex).index()) {
	            	phraseIndex++;
	            }
	            compoundPhraseList.add(phraseIndex, compound);
	         }
	    	String compoundPhrase = "";
	    	compoundPhrase += compoundPhraseList.get(0).toString();
	    	for(int i = 1; i < compoundPhraseList.size(); i++) {
	    		compoundPhrase += " ";
	    		compoundPhrase += compoundPhraseList.get(i).toString();
	    	}
	    	topicPhrases.add(compoundPhrase);
	    	
	    	
		    List<Tree> leaves = claimConstituencyParse.getLeaves();
		    Tree parseTree = null;
		    parseTree = leaves.get(topic.index()-1);
	
		    //return ArrayList of subject phrase, from broadest to most narrow
		    Tree topicTree = parseTree.deepCopy();
		    ArrayList<Tree> topicPhraseTree = new ArrayList<Tree>();
		    topicPhraseTree.add(topicTree);
		    String[] nouns = {"NN", "NNS", "NP", "NPS"};
		    while(parseTree.parent(claimConstituencyParse) != null) {
		    	parseTree = parseTree.parent(claimConstituencyParse);
		    	if(Arrays.asList(nouns).contains(parseTree.label().toString())) {
		    		topicTree = parseTree.deepCopy();
		    		topicPhraseTree.add(0, topicTree);
		    		
		    	}
		    }
		    
		    for (int i = 0; i < topicPhraseTree.size(); i++) {
			    List<Tree> topicWords = topicPhraseTree.get(i).getLeaves();
			    String topicPhrase = topicWords.get(0).toString();
			    for(int j = 1; j < topicWords.size(); j++) {
			    	boolean endsInPossessive = (j == topicWords.size()-1) && topicWords.get(j).toString().equals("\'s");
			    	if(!endsInPossessive) {
				    	topicPhrase += " ";
				    	topicPhrase += topicWords.get(j).toString();
			    	}
			    }
			    if(topicPhrase.startsWith("the ") || topicPhrase.startsWith("The ")) {
			    	ArrayList<String> namedEntities = getNamedEntities(claimSentence, pipeline);
			    	if (!namedEntities.contains(topicPhrase)){
			    		topicPhrase = topicPhrase.substring(4);
			    	}
			    	
			    }
			    topicPhrases.add(topicPhrase);
		    }
	    }
	    //get rid of duplicates
	    topicPhrases = (ArrayList<String>) topicPhrases.stream().distinct().collect(Collectors.toList());
		return topicPhrases;
	}
	
	private static IndexedWord getDescendentWithReln(SemanticGraph dependencyTree, IndexedWord vertex, GrammaticalRelation reln, int layer) {
		IndexedWord descendent = dependencyTree.getChildWithReln(vertex, reln);
		if(descendent == null) {
			Set<IndexedWord> children = dependencyTree.getChildren(vertex);
			for(IndexedWord child: children) {
				if(descendent == null && layer >= 0) {
					descendent = getDescendentWithReln(dependencyTree, child, reln, layer-1);
				}
			}
		}
		
		return descendent;
	}
	
	private static ArrayList<Integer> getTextMatches(String sentence, String text){
		ArrayList<String> matchingSentences = new ArrayList<String>();
		ArrayList<Integer> matchingSentenceNums = new ArrayList<Integer>();
		String[] textSentences = text.split(" [.] ");
		for(int i = 0; i < textSentences.length; i++) {
			if(textSentences[i].contains(sentence)) {
				matchingSentences.add(textSentences[i]);
				matchingSentenceNums.add(i);
			}
			//negates
			String negateSentence = textSentences[i].replaceAll("not ", "").replaceAll("n\'t", "").replaceAll("never ", "");
			if(negateSentence.contains(sentence)) {
				matchingSentences.add(textSentences[i]);
				matchingSentenceNums.add(i);
			}
		}
		
		return matchingSentenceNums;
	}
	
	//possibly remove subject of article from included named entities?
	private static ArrayList<Integer> getNamedEntityMatches(String sentence, String text, StanfordCoreNLP pipeline){
		ArrayList<String> namedEntities = getNamedEntities(sentence, pipeline);
		ArrayList<String> matchingSentences = new ArrayList<String>();
		ArrayList<Integer> matchingSentenceNums = new ArrayList<Integer>();
		String[] textSentences = text.split(" [.] ");
		for(int i = 0; i < textSentences.length; i++) {
			//normalize sentence for matching
			String normalSent = textSentences[i].replaceAll(" , ", ", ").replaceAll(" : ", ": ").replaceAll(" ; ", "; ").replaceAll(" '", "'").replaceAll(" -- ", "–").replaceAll(" -LRB- ", " (").replaceAll("-RRB- ", ") ").replaceAll("-LSB- ", " [").replaceAll("-RSB- ", "] ");
			boolean includesAll = true;
			for(int j = 0; j < namedEntities.size(); j++) {
				if(!normalSent.contains(namedEntities.get(j))) {
					includesAll = false;
				}
			}
			if(includesAll) {
				matchingSentences.add(textSentences[i]);
				matchingSentenceNums.add(i);
			}
		}
		
		return matchingSentenceNums;
	}
	
	private static ArrayList<String> getNamedEntities(String sentence, StanfordCoreNLP pipeline){
		ArrayList<CoreLabel> tokens = getSentenceTokens(sentence, pipeline);
		ArrayList<String> namedEntities = new ArrayList<String>();
		String namedEntity = "";
		boolean neActive = false;
		for(int i = 0; i < tokens.size(); i++) {
			String ne = tokens.get(i).get(NamedEntityTagAnnotation.class);
			if(!neActive && !ne.equals("O")) {
				neActive = true;
				namedEntity += tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && !ne.equals("O")) {
				namedEntity += " " + tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && ne.equals("O")) {
				neActive = false;
				namedEntities.add(namedEntity);
				namedEntity = "";
			}
		}
		if(neActive) {
			namedEntities.add(namedEntity);
		}
		return namedEntities;
	}
	
	private static ArrayList<CoreLabel> getSentenceTokens(String sentence, StanfordCoreNLP pipeline){
		Annotation document = new Annotation(sentence);
	    pipeline.annotate(document);
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    CoreMap tokenSen = sentences.get(0);
	    
	    ArrayList<CoreLabel> tokens = new ArrayList<CoreLabel>();
	    for (CoreLabel token: tokenSen.get(TokensAnnotation.class)) {
	        tokens.add(token);
	      }
	    return tokens;
	}
	
	private static Map<String, String> getDocFromTitle(Map<String, Object[]> wikiMap, String subject) {
		//System.out.println("getDocBegin");
		String doc = "";
		String text = "";
		Map<String, String> wikiDoc = new HashMap<String, String>();
		String urlTitle = subject.replace(' ', '_');
		//char urlFirstLetter = urlTitle.charAt(0);
		//File wikiDir = new File(wikiDirName);
		//File[] wikiEntries = wikiDir.listFiles();
		String wikiEntry = "";
		if (!subject.equals("No subject detected in sentence") && wikiMap.containsKey(urlTitle)){
			Object[] fileInfo = wikiMap.get(urlTitle);
			//System.out.println("filename: " + fileInfo[0]);
			String wikiListName = wikiDirName + "\\" + fileInfo[0];
			File wikiList = new File(wikiListName);
			try {
				Scanner s = new Scanner(wikiList);
				wikiEntry = s.nextLine();
				JSONObject wikiJson = new JSONObject(wikiEntry);
				while(s.hasNext() && doc.isEmpty()) {
				    wikiEntry = s.nextLine();
				    wikiJson = new JSONObject(wikiEntry);
				    if(wikiJson.getString("id").equalsIgnoreCase(urlTitle)) {
				    	doc = wikiJson.getString("id");
				    	text = wikiJson.getString("text");
				    	if (text.contains((subject+" may refer to: "))) {
				    		text = "DISAMBIGUATION";
				    	}
				    	wikiDoc.put("id", doc);
				    	wikiDoc.put("text", text);
				    }
				}	

					
				s.close();
			} catch (FileNotFoundException e) {
				System.out.println("Could not open file  "+fileInfo[0]);
				e.printStackTrace();
			} catch (JSONException e) {
				System.out.println("There was a problem with the json from " + wikiEntry);
				e.printStackTrace();
			}
//			for (File wikiEntryList : wikiEntries) {
//				try {
//					Scanner s = new Scanner(wikiEntryList);
//					if(s.hasNext()) {
//						wikiEntry = s.nextLine();
//						JSONObject wikiJson = new JSONObject(wikiEntry);
//						String id = wikiJson.getString("id");
//						if(!id.isEmpty()) {
//							char firstLetter = id.charAt(0);
//							//wiki entries are partially in alphabetical order. this skips documents not close to desired word.
//							if(firstLetter >= urlFirstLetter-1 && firstLetter <= urlFirstLetter+1) {
//								if(wikiJson.getString("id").equalsIgnoreCase(urlTitle)) {
//							    	doc = wikiJson.getString("id");
//							    	text = wikiJson.getString("text");
//							    	wikiDoc.put("id", doc);
//							    	wikiDoc.put("text", text);
//							    }
//								while(s.hasNext() && doc.isEmpty()) {
//								    wikiEntry = s.nextLine();
//								    wikiJson = new JSONObject(wikiEntry);
//								    if(wikiJson.getString("id").equalsIgnoreCase(urlTitle)) {
//								    	doc = wikiJson.getString("id");
//								    	text = wikiJson.getString("text");
//								    	wikiDoc.put("id", doc);
//								    	wikiDoc.put("text", text);
//								    }
//								}	
//							}
//						}
//					}
//						
//					s.close();
//				} catch (FileNotFoundException e) {
//					System.out.println("Could not open file  "+wikiEntryList.getName());
//					e.printStackTrace();
//				} catch (JSONException e) {
//					System.out.println("There was a problem with the json from " + wikiEntry);
//					e.printStackTrace();
//				}
//			}
		}
		
		return wikiDoc;
	}
	
	private static JSONObject convertToJSON(String claimID, String claim, String mainDoc, ArrayList<Integer> matchingSentenceNums, ArrayList<String> matchingSentences) {
		Map<String, String> jsonMap = new HashMap<String, String>();
		jsonMap.put("id", claimID);
		jsonMap.put("claim", claim);
		jsonMap.put("wiki id", mainDoc);
		//jsonMap.put("evidence", String.join(",", matchingSentenceNums));
		jsonMap.put("supporting sentences", String.join("\", \"", matchingSentences));
		return new JSONObject(jsonMap);
	}
	
	public static Map<String, Object[]> getWikiMap(String wikiDirName) {
		Map <String, Object[]> wikiMap = new HashMap<String, Object[]>();
		
		File wikiDir = new File(wikiDirName);
		File[] wikiEntries = wikiDir.listFiles();
		String wikiEntry = "";
		int filesProcessed = 0;
		if (wikiEntries != null){
			for (File wikiEntryList : wikiEntries) {
				try {
					Scanner s = new Scanner(wikiEntryList);
					int wikiLine = 0;

						while(s.hasNext()) {
						    wikiEntry = s.nextLine();
						    JSONObject wikiJson = new JSONObject(wikiEntry);
						    String id = wikiJson.getString("id");
						    if(!id.isEmpty()) {
						    	String fileName = wikiEntryList.getName();
						    	Object[] docLocation = new Object[2];
						    	docLocation[0] = fileName;
						    	docLocation[1] = wikiLine;
								wikiMap.put(id, docLocation);
						    }
						    wikiLine++;
						}
						filesProcessed++;
						if(filesProcessed%10 == 0) {
							System.out.println("Wiki processing "+filesProcessed+"% done.");
						}
						
					s.close();
				} catch (FileNotFoundException e) {
					System.out.println("Could not open file  "+wikiEntryList.getName());
					e.printStackTrace();
				} catch (JSONException e) {
					System.out.println("There was a problem with the json from " + wikiEntry);
					e.printStackTrace();
				}
			}
		}
		return wikiMap;
	}
	


}

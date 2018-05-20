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

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;


public class DocFinder {
	static String statementsFileName = "shared_task_dev_public.jsonl";
	static String resultsFileName = "results.jsonl";
	static String wikiDirName = "wiki-dump";
	static int numClaimsToTest = 100;
	
	
	
	public static void main(String[] args) {
		GrammaticalRelation[] subjectRelations= {UniversalEnglishGrammaticalRelations.SUBJECT, UniversalEnglishGrammaticalRelations.CLAUSAL_PASSIVE_SUBJECT, 
	    		UniversalEnglishGrammaticalRelations.CLAUSAL_SUBJECT, UniversalEnglishGrammaticalRelations.CONTROLLING_CLAUSAL_PASSIVE_SUBJECT,
	    		UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT, UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT};
		GrammaticalRelation[] objectRelations= {UniversalEnglishGrammaticalRelations.DIRECT_OBJECT,UniversalEnglishGrammaticalRelations.OBJECT};
		
		String statement = "";
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,parse,depparse");
	    props.setProperty("coref.algorithm", "neural");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		try {
			Scanner statementReader = new Scanner(new FileReader(statementsFileName));
			File oldFile = new File(resultsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFileName, true));
			int statementCount =0;
			while(statementReader.hasNext() && statementCount < numClaimsToTest) {
			    statement = statementReader.nextLine();
			    JSONObject stateJson = new JSONObject(statement);
			    String claimID = stateJson.get("id").toString();
			    String claim = stateJson.getString("claim");
			    ArrayList<String> claimSubjects = getClaimTopic(claim, subjectRelations, pipeline, false);
			    String mainDoc = "";
			    String claimSubject = "";
			    int i =0;
			    while(mainDoc.isEmpty() && i < claimSubjects.size()) {
			    	claimSubject = claimSubjects.get(i);
			    	mainDoc = getDocFromTitle(claimSubject);
			    	i++;
			    }
			    if (mainDoc.isEmpty()) {
			    	ArrayList<String> claimSubjectDependents = getClaimTopic(claim, subjectRelations, pipeline, true);
				    String claimSubjectDependent = "";
				    i =0;
				    while(mainDoc.isEmpty() && i < claimSubjectDependents.size()) {
				    	claimSubjectDependent = claimSubjectDependents.get(i);
				    	mainDoc = getDocFromTitle(claimSubjectDependent);
				    	i++;
				    }
			    }
			    if (mainDoc.isEmpty()) {
			    	ArrayList<String> claimObjects = getClaimTopic(claim, objectRelations, pipeline, false);
				    String claimObject = "";
				    i =0;
				    while(mainDoc.isEmpty() && i < claimObjects.size()) {
				    	claimObject = claimObjects.get(i);
				    	mainDoc = getDocFromTitle(claimObject);
				    	i++;
				    }
			    } 
			    if (mainDoc.isEmpty()) {
			    	ArrayList<String> claimObjectDependents = getClaimTopic(claim, objectRelations, pipeline, true);
				    String claimObjectDependent = "";
				    i =0;
				    while(mainDoc.isEmpty() && i < claimObjectDependents.size()) {
				    	claimObjectDependent = claimObjectDependents.get(i);
				    	mainDoc = getDocFromTitle(claimObjectDependent);
				    	i++;
				    }
			    }
			    if (mainDoc.isEmpty()) {
			    	System.out.println("unable to find "+claimSubject+" via title. Must find via content.");
			    	//mainDoc = getDocFromContent(claimSubject);
			    	mainDoc = "no subject found";
			    }
			    JSONObject result = convertToJSON(claimID, claim, mainDoc);
			    writer.append(result.toString());
			    writer.append("\n");
			    statementCount++;
			    DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
			    LocalDateTime now = LocalDateTime.now();  
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
		    		//topic = claimDependencyParse.getChildWithReln(topicParent, dependencyRelations[dependencyRelation]);
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
	    
	    Tree claimConstituencyParse = claimSen.constituencyParse();
	    if(topics.isEmpty()) {
	    	ArrayList<String> nullReturn = new ArrayList<String>();
	    	nullReturn.add("No subject detected in sentence");
	    	return nullReturn;
	    }
	    
	    ArrayList<String> topicPhrases = new ArrayList<String>();
	    for(int h = 0; h < topics.size(); h++) {
	    	topic = topics.get(0);
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
			    	if(!(j == topicWords.size()-1 && topicWords.get(j).toString().equals("\'s"))) {
				    	topicPhrase += " ";
				    	topicPhrase += topicWords.get(j).toString();
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
	
	
	private static String getDocFromTitle(String subject) {
		String doc = "";
		String urlTitle = subject.replace(' ', '_');
		char urlFirstLetter = urlTitle.charAt(0);
		File wikiDir = new File(wikiDirName);
		File[] wikiEntries = wikiDir.listFiles();
		String wikiEntry = "";
		if (wikiEntries != null && !subject.equals("No subject detected in sentence")){
			for (File wikiEntryList : wikiEntries) {
				try {
					Scanner s = new Scanner(wikiEntryList);
					if(s.hasNext()) {
						wikiEntry = s.nextLine();
						JSONObject wikiJson = new JSONObject(wikiEntry);
						String id = wikiJson.getString("id");
						if(!id.isEmpty()) {
							char firstLetter = id.charAt(0);
							//wiki entries are partially in alphabetical order. this skips documents not close to desired word.
							if(firstLetter >= urlFirstLetter-1 && firstLetter <= urlFirstLetter+1) {
								if(wikiJson.getString("id").equalsIgnoreCase(urlTitle)) {
							    	doc = wikiJson.getString("id");
							    }
								while(s.hasNext() && doc.isEmpty()) {
								    wikiEntry = s.nextLine();
								    wikiJson = new JSONObject(wikiEntry);
								    if(wikiJson.getString("id").equalsIgnoreCase(urlTitle)) {
								    	doc = wikiJson.getString("id");
								    }
								}	
							}
						}
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
		
		return doc;
	}
	
	private static JSONObject convertToJSON(String claimID, String claim, String mainDoc) {
		Map<String, String> jsonMap = new HashMap<String, String>();
		jsonMap.put("id", claimID);
		jsonMap.put("claim", claim);
		jsonMap.put("found doc", mainDoc);
		return new JSONObject(jsonMap);
	}
	
//	private static String getDocFromContent(String subject) {
//		String doc = "";
//		int subjectCount = 0;
//		File wikiDir = new File(wikiDirName);
//		File[] wikiEntries = wikiDir.listFiles();
//		String wikiEntry = "";
//		if (wikiEntries != null){
//			for (File wikiEntryList : wikiEntries) {
//				try {
//					Scanner s = new Scanner(wikiEntryList);
//					while(s.hasNext() && doc.isEmpty()) {
//						wikiEntry = s.nextLine();
//					    JSONObject wikiJson = new JSONObject(wikiEntry);
//					    String wikiText = wikiJson.getString("text");
//					    int count = 0;
//					    int lastIndex = 0;
//					    while(lastIndex != -1) {
//					    	lastIndex = wikiText.indexOf(subject);
//					    	if(lastIndex != -1){
//					            count++;
//					            lastIndex += subject.length();
//					        }
//					    }
//					    if(count > subjectCount) {
//					    	subjectCount = count;
//					    	doc = wikiJson.getString("id");
//					    }
//					}	
//					s.close();
//				} catch (FileNotFoundException e) {
//					System.out.println("Could not open file  "+wikiEntryList.getName());
//					e.printStackTrace();
//				} catch (JSONException e) {
//					System.out.println("There was a problem with the json from " + wikiEntry);
//					e.printStackTrace();
//				}
//			}
//		} else {
//		    System.out.println("Something went wrong with the wiki file directory at "+wikiDirName);
//		}
//		
//		
//		return doc;
//	}
	


}

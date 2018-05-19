import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;

import org.json.*;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;


public class DocFinder {
	static String statementsFileName = "shared_task_dev_public.jsonl";
	static String resultsFileName = "results.jsonl";
	static String wikiDirName = "wiki-dump";
	
	public static void main(String[] args) {
		String statement = "";
		try {
			Scanner statementReader = new Scanner(new FileReader(statementsFileName));
			File oldFile = new File(resultsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFileName, true));
			while(statementReader.hasNext()) {
			    statement = statementReader.nextLine();
			    JSONObject stateJson = new JSONObject(statement);
			    String claimID = stateJson.get("id").toString();
			    String claim = stateJson.getString("claim");
			    String claimSubject = getClaimSubject(claim);
			    String mainDoc = getDocFromTitle(claimSubject);
			    if (mainDoc.isEmpty()) {
			    	mainDoc = getDocFromContent(claimSubject);
			    }
			    JSONObject result = convertToJSON(claimID, claim, mainDoc);
			    writer.append(result.toString());
			    writer.append("\n");
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
	
	
	private static String getClaimSubject(String claimSentence) {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,parse,depparse");
	    props.setProperty("coref.algorithm", "neural");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    CoreDocument document = new CoreDocument(claimSentence);
	    pipeline.annotate(document);
	    CoreSentence claimSen = document.sentences().get(0);
	    SemanticGraph claimDependencyParse = claimSen.dependencyParse();
	    IndexedWord subject = claimDependencyParse.getChildWithReln(claimDependencyParse.getFirstRoot(), UniversalEnglishGrammaticalRelations.SUBJECT);
	    if(subject == null) {
	    	subject = claimDependencyParse.getChildWithReln(claimDependencyParse.getFirstRoot(), UniversalEnglishGrammaticalRelations.CLAUSAL_PASSIVE_SUBJECT);
	    }
	    if(subject == null) {
	    	subject = claimDependencyParse.getChildWithReln(claimDependencyParse.getFirstRoot(), UniversalEnglishGrammaticalRelations.CONTROLLING_CLAUSAL_PASSIVE_SUBJECT);
	    }
	    if(subject == null) {
	    	subject = claimDependencyParse.getChildWithReln(claimDependencyParse.getFirstRoot(), UniversalEnglishGrammaticalRelations.CONTROLLING_NOMINAL_PASSIVE_SUBJECT);
	    }
	    if(subject == null) {
	    	subject = claimDependencyParse.getChildWithReln(claimDependencyParse.getFirstRoot(), UniversalEnglishGrammaticalRelations.CONTROLLING_NOMINAL_SUBJECT);
	    }
	    if(subject == null) {
	    	subject = claimDependencyParse.getChildWithReln(claimDependencyParse.getFirstRoot(), UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT);
	    }
	    if(subject == null) {
	    	subject = claimDependencyParse.getChildWithReln(claimDependencyParse.getFirstRoot(), UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT);
	    }
	    Tree claimConstituencyParse = claimSen.constituencyParse();
	    List<Tree> leaves = claimConstituencyParse.getLeaves();
	    Tree parseTree = leaves.get(subject.index()-1);
	    Tree subjTree = parseTree.deepCopy();
	    String[] nouns = {"NN", "NNS", "NP", "NPS"};
	    while(parseTree.parent(claimConstituencyParse) != null) {
	    	parseTree = parseTree.parent(claimConstituencyParse);
	    	if(Arrays.asList(nouns).contains(parseTree.label().toString())) {
	    		subjTree = parseTree.deepCopy();
	    	}
	    }
	    List<Tree> subjWords = subjTree.getLeaves();
	    String subjectPhrase = subjWords.get(0).toString();
	    for(int i = 1; i < subjWords.size(); i++) {
	    	subjectPhrase += " ";
	    	subjectPhrase += subjWords.get(i).toString();
	    }
	    
		
		return subjectPhrase;
	}
	
	private static String getDocFromTitle(String subject) {
		String doc = "";
		String urlTitle = subject.replace(' ', '_');
		File wikiDir = new File(wikiDirName);
		File[] wikiEntries = wikiDir.listFiles();
		String wikiEntry = "";
		if (wikiEntries != null){
			for (File wikiEntryList : wikiEntries) {
				try {
					Scanner s = new Scanner(wikiEntryList);
					while(s.hasNext() && doc.isEmpty()) {
					    wikiEntry = s.nextLine();
					    JSONObject wikiJson = new JSONObject(wikiEntry);
					    if(wikiJson.getString("id").equals(urlTitle)) {
					    	doc = wikiJson.getString("id");
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
		} else {
		    System.out.println("Something went wrong with the wiki file directory at "+wikiDirName);
		}
		
		
		return doc;
	}
	
	private static String getDocFromContent(String subject) {
		String doc = "";
		int subjectCount = 0;
		File wikiDir = new File(wikiDirName);
		File[] wikiEntries = wikiDir.listFiles();
		String wikiEntry = "";
		if (wikiEntries != null){
			for (File wikiEntryList : wikiEntries) {
				try {
					Scanner s = new Scanner(wikiEntryList);
					while(s.hasNext() && doc.isEmpty()) {
						wikiEntry = s.nextLine();
					    JSONObject wikiJson = new JSONObject(wikiEntry);
					    String wikiText = wikiJson.getString("text");
					    int count = 0;
					    int lastIndex = 0;
					    while(lastIndex != -1) {
					    	lastIndex = wikiText.indexOf(subject);
					    	if(lastIndex != -1){
					            count++;
					            lastIndex += subject.length();
					        }
					    }
					    if(count > subjectCount) {
					    	subjectCount = count;
					    	doc = wikiJson.getString("id");
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
		} else {
		    System.out.println("Something went wrong with the wiki file directory at "+wikiDirName);
		}
		
		
		return doc;
	}
	
	private static JSONObject convertToJSON(String claimID, String claim, String mainDoc) {
		Map<String, String> jsonMap = new HashMap<String, String>();
		jsonMap.put("id", claimID);
		jsonMap.put("claim", claim);
		jsonMap.put("evidence", mainDoc);
		return new JSONObject(jsonMap);
	}

}

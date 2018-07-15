import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.HashMap;
//import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
//import java.util.Set;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

//import com.google.common;
//import com.google.common.reflect.TypeToken;
import java.lang.reflect.Type;
import java.text.Normalizer;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Tree;

public class DocFinderTester {
		
	static String answersFileName = "shared_task_dev.jsonl";
	static String resultsFileName = "results.jsonl";
	static String analysisFileName = "analysis.jsonl";
	static int numClaimsTested = 200;
	
	public static void main(String[] args) {
		try {
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize,ssplit,pos,parse,depparse");
		    props.setProperty("coref.algorithm", "neural");
		    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
			Scanner answersReader = new Scanner(new FileReader(answersFileName));
			Scanner resultsReader = new Scanner(new FileReader(resultsFileName));
			File oldFile = new File(analysisFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(analysisFileName, true));
			int countCorrect = 0;
			int countNull = 0;
			
			int totalWikis = 0;
			int backups = 0;
			writer.append("Erroneous retrieved wiki articles:\n\n");
			while(resultsReader.hasNext()) {
				String result = Normalizer.normalize(resultsReader.nextLine(), Normalizer.Form.NFD);
				String answer = Normalizer.normalize(answersReader.nextLine(), Normalizer.Form.NFD);
				JSONObject resultJson = new JSONObject(result);
				JSONObject answerJson = new JSONObject(answer);
				//JSONObject wikiJson = (JSONObject) resultJson.get("wiki info");
				String wikiJson = resultJson.get("wiki info").toString();
				Type wikiMapType = new TypeToken<Map<String, Map<String, Object>>>(){}.getType();  
				Map<String, Object> wikiInfo = new Gson().fromJson(wikiJson, wikiMapType);
				JSONArray backupWikiJSON = (JSONArray) resultJson.get("backup wikis");
				totalWikis += wikiInfo.size();
				backups += backupWikiJSON.length();
				
				//Set<String> docsFound = new HashSet<String>();
//				for(String wiki: wikiJson.keySet()) {
//					
//				}
//				
//				 = wikiJson.keys();
				String claim = (String) resultJson.get("claim");
				JSONArray answerEvidence = (JSONArray) answerJson.get("evidence");
				JSONArray answerEvidence1 = (JSONArray) answerEvidence.get(0);
				JSONArray answerEvidence2 = (JSONArray) answerEvidence1.get(0);
				Object answerWiki =  answerEvidence2.get(2);
				ArrayList<String> backupWikis = new ArrayList<String>();
				for(int i = 0; i < backupWikiJSON.length(); i++){
					backupWikis.add(backupWikiJSON.getString(i));
				}
				if(answerWiki.equals(null)) {
					countNull++;
				}
				else if(wikiInfo.keySet().contains(answerWiki) || backupWikis.contains(answerWiki.toString().toLowerCase())) {
					countCorrect++;
				}
				else {
					writer.append("Claim: \"" + claim+"\"\n");
					writer.append("Correct document: " + answerWiki + "\n");
					writer.append("Found Documents: \n");
					for(String wiki: wikiInfo.keySet()) {
						writer.append(wiki + "\n");
					}
					writer.append("Backup Documents: \n");
					for(int i = 0; i < backupWikis.size(); i++) {
						writer.append(backupWikis.get(i) + "\n");
					}
					
					String claimSentence = claim;
					CoreDocument document = new CoreDocument(claimSentence);
				    pipeline.annotate(document);
				    CoreSentence claimSen = document.sentences().get(0);
				    SemanticGraph claimDependencyParse = claimSen.dependencyParse();
				    Tree claimConstituencyParse = claimSen.constituencyParse();
				    writer.append("Constituency Tree: " + claimConstituencyParse+"\n");
				    writer.append("Dependency Graph: " + claimDependencyParse+"\n\n");
				    
				}
				

			}
			writer.append("Correct documents found: " + countCorrect+"/"+(numClaimsTested-countNull)+"\n");
			writer.append("Total primary documents found: " + totalWikis+"\n");
			writer.append("Backup documents found: " + backups+"\n");
			answersReader.close();
			resultsReader.close();
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (JSONException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	
}

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Properties;
import java.util.Scanner;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Tree;

public class DocFinderTester {
		
	static String answersFileName = "shared_task_dev.jsonl";
	static String resultsFileName = "results.jsonl";
	static String analysisFileName = "analysis.jsonl";
	static int numClaimsTested = 100;
	
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
			writer.append("Erroneous retrieved wiki articles:\n\n");
			while(resultsReader.hasNext()) {
				String result = resultsReader.nextLine();
				String answer = answersReader.nextLine();
				JSONObject resultJson = new JSONObject(result);
				JSONObject answerJson = new JSONObject(answer);
				String resultWiki = (String) resultJson.get("found doc");
				String claim = (String) resultJson.get("claim");
				JSONArray answerEvidence = (JSONArray) answerJson.get("evidence");
				JSONArray answerEvidence1 = (JSONArray) answerEvidence.get(0);
				JSONArray answerEvidence2 = (JSONArray) answerEvidence1.get(0);
				Object answerWiki =  answerEvidence2.get(2);
				if(answerWiki.equals(null)) {
					countNull++;
				}
				else if(answerWiki.equals(resultWiki)) {
					countCorrect++;
				}
				else {
					if(resultWiki.equals("no subject found")) {
						resultWiki = "Unable to find any wiki";
					};
					writer.append("Claim: \"" + claim+"\"\n");
					writer.append("Found Document: \"" + resultWiki+"\"\n");
					writer.append("Correct Document: \"" + answerWiki+"\"\n");
					
					
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
			writer.append("Correctly found wikis (omit null correct answer): " + countCorrect+"/"+(numClaimsTested-countNull)+"\n");
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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Tree;

public class SentenceFinderTester {
	
	static String answersFileName = "shared_task_dev.jsonl";
	static String resultsFileName = "sentence_results.jsonl";
	static String analysisFileName = "sentence_analysis.jsonl";
	static int numClaimsTested = 100;
	
	public static void main(String[] args) {
		try {
//			Properties props = new Properties();
//			props.setProperty("annotators", "tokenize,ssplit,pos,parse,depparse");
//		    props.setProperty("coref.algorithm", "neural");
//		    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
			Scanner answersReader = new Scanner(new FileReader(answersFileName));
			Scanner resultsReader = new Scanner(new FileReader(resultsFileName));
			File oldFile = new File(analysisFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(analysisFileName, true));
			int countCorrect = 0;
			int countWrong = 0;
			int countSentenceWrong = 0;
			int countMissed = 0;
			int claimCount = 0;
			while(resultsReader.hasNext()) {
				claimCount++;
				String answer = Normalizer.normalize(answersReader.nextLine(), Normalizer.Form.NFD);	
				JSONObject answerJson = new JSONObject(answer);
				JSONArray answerEvidence = (JSONArray) answerJson.get("evidence");
				ArrayList<Object[]> correctEvidence = new ArrayList<Object[]>();
				for(int i = 0; i < answerEvidence.length(); i++) {
					JSONArray evidenceSet = answerEvidence.getJSONArray(i);
					JSONArray primarySentence = evidenceSet.getJSONArray(0);
					if(!primarySentence.get(2).equals(null)) {
						String wikiName = primarySentence.get(2).toString();
						Integer sentNum = primarySentence.getInt(3);
						Object[] answerArr = {wikiName, sentNum};
						if(!ArrayIsInList(correctEvidence, answerArr)) {
							correctEvidence.add(answerArr);
						}
					}	
					
				}
				
				String result = resultsReader.nextLine();
				JSONObject resultJson = new JSONObject(result);
				String claim = Normalizer.normalize(resultJson.getString("claim"), Normalizer.Form.NFC);
				String label = resultJson.getString("label");
				JSONArray resultEvidence = (JSONArray) resultJson.get("evidence");
				JSONArray wikiLines = (JSONArray) resultJson.get("sentences");
				ArrayList<Object[]> foundEvidence = new ArrayList<Object[]>();
				for(int i = 0; i < resultEvidence.length(); i++) {
					JSONArray primarySentence = resultEvidence.getJSONArray(i);
					String wikiName = Normalizer.normalize(primarySentence.get(0).toString(), Normalizer.Form.NFC);
					Integer sentNum = primarySentence.getInt(2);
					Object[] resultArr = {wikiName, sentNum};
					foundEvidence.add(resultArr);
				}
				
				boolean found = false;
				ArrayList<Object[]> correctSentences = new ArrayList<Object[]>();
				ArrayList<Object[]> wrongSentences = new ArrayList<Object[]>();
				int oldCountWrong = countWrong;
				if(correctEvidence.isEmpty() && foundEvidence.isEmpty()) {
					countCorrect++;
				}
				else {
					for(Object[] foundSent : foundEvidence) {
						if(ArrayIsInList(correctEvidence, foundSent) && !found) {
							correctSentences.add(foundSent);
							countCorrect++;
							found = true;
						}
						else if(ArrayIsInList(correctEvidence, foundSent)) {
							correctSentences.add(foundSent);
						}
						else if(!ArrayIsInList(correctEvidence, foundSent)) {
							countWrong++;
							wrongSentences.add(foundSent);
						}
					}
				}
				if(!found && correctEvidence.size() > 0) {
					countMissed++;
				}
				if(countWrong > oldCountWrong) {
					countSentenceWrong++;
				}
				
				writer.append("\n\n\n\nClaim " + claimCount + ": " + claim + "\n");
				writer.append("Label: " + label + "\n");
//				writer.append("Relevant words: " + relevantWords + "\n");
				if(wikiLines.length() == 0 && !label.equals("NOT ENOUGH INFO")) {
					writer.append("No single evidence sets \n");
					continue;
				}
				if(correctEvidence.size() > 0) {
					writer.append("Wiki title: " + correctEvidence.get(0)[0] + "\n");
				}
				writer.append("\nCorrect sentences in training set: \n");
				for(Object[] answerSent : correctEvidence) {
					writer.append(wikiLines.getString((int) answerSent[1]) + "\n");
				}
				writer.append("\nCorrectly found sentences: \n");
				for(Object[] resultSent : correctSentences) {
					writer.append(wikiLines.getString((int) resultSent[1]) + "\n");
				}
				writer.append("\nIncorrectly found sentences: \n");
				for(Object[] resultSent : wrongSentences) {
					writer.append(wikiLines.getString((int) resultSent[1]) + "\n");
				}
				

			}
			writer.append("\n\n\n\n");
			writer.append("Number of claims with a correct sentence found (or it correctly found none): " + countCorrect+"/"+numClaimsTested+"\n");
			writer.append("Number of sentences where none of the correct sentences were found: " + countMissed+"/"+numClaimsTested+"\n");
			writer.append("Number of incorrect sentences found: " + countWrong+"\n");
			writer.append("Number of claims with incorrect sentences found: " + countSentenceWrong+"/"+numClaimsTested+"\n");
			
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
	
	private static boolean ArrayIsInList(ArrayList<Object[]> list, Object[] arr) {
		boolean inside = false;
		for(Object[] listArr : list) {
			if(listArr[0].equals(arr[0]) && listArr[1].equals(arr[1])) {
				inside = true;
			}
		}
		return inside;
	}
}

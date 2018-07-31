import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class SentenceFinderTester {
	static String answersFileName = "shared_task_dev.jsonl";
	static String resultsFileName = "found_sentences.jsonl";
	static String analysisFileName = "sentence_analysis.jsonl";
	
	public static void main(String[] args) {
		try {
			Scanner answersReader = new Scanner(new FileReader(answersFileName));
			Scanner resultsReader = new Scanner(new FileReader(resultsFileName));
			File oldFile = new File(analysisFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(analysisFileName, true));
			double countCorrect = 0;
			double countWrong = 0;
			int countClaimCorrect = 0;
			int countClaimWrong = 0;
			int countClaimMissed = 0;
			double countMissed = 0;
			int claimCount = 0;
			int verifiableClaimCount = 0;

			int verifiableClaimWithNoEvidenceFound = 0;

			// sum of precisions of all claims
			double precisionSum = 0;
            // sum of recalls of all claims
			double recallSum = 0;
			while(resultsReader.hasNext()) {
				claimCount++;
				double precision = 0;
				double recall = 0;
				String answer = Normalizer.normalize(answersReader.nextLine(), Normalizer.Form.NFC);	
				JSONObject answerJson = new JSONObject(answer);
				JSONArray answerEvidence = (JSONArray) answerJson.get("evidence");
				String label = answerJson.getString("label");
				int goldEvidenceCount = answerEvidence.length();
				ArrayList<Object[]> correctEvidence = new ArrayList<Object[]>();
				for(int i = 0; i < answerEvidence.length(); i++) {
					JSONArray evidenceSet = answerEvidence.getJSONArray(i);
					JSONArray primarySentence = evidenceSet.getJSONArray(0);
					if(!primarySentence.get(2).equals(null) && evidenceSet.length() == 1) {
						String wikiName = primarySentence.get(2).toString();
						Integer sentNum = primarySentence.getInt(3);
						Object[] answerArr = {wikiName, sentNum};
						if(!ArrayIsInList(correctEvidence, answerArr)) {
							correctEvidence.add(answerArr);
						}
					}
					else if(!primarySentence.get(2).equals(null) && evidenceSet.length() != 1) {
						//multi sentence answer, which we always miss
						Object[] answerArr = {"multi", 0};
						correctEvidence.add(answerArr);
					}
				}

				String result = resultsReader.nextLine();
				JSONObject resultJson = new JSONObject(result);
				String claim = Normalizer.normalize(resultJson.getString("claim"), Normalizer.Form.NFC);
				
				JSONArray resultEvidence = (JSONArray) resultJson.get("evidence");

				ArrayList<Object[]> foundEvidence = new ArrayList<Object[]>();
				for(int i = 0; i < resultEvidence.length(); i++) {
					JSONArray resultSet = resultEvidence.getJSONArray(i);
					JSONArray primarySentence = resultSet.getJSONArray(0);
					if(!primarySentence.get(2).equals(null)) {
						String wikiName = primarySentence.get(0).toString();
						Integer sentNum = primarySentence.getInt(1);
						Object[] answerArr = {wikiName, sentNum};
						if(!ArrayIsInList(foundEvidence, answerArr)) {
							foundEvidence.add(answerArr);
						}
					}
				}

				ArrayList<Object[]> correctSentences = new ArrayList<Object[]>();
				ArrayList<Object[]> wrongSentences = new ArrayList<Object[]>();
				ArrayList<Object[]> missedSentences = new ArrayList<Object[]>();

				boolean correct = false;
				boolean wrong = false;
				for(Object[] foundSent : foundEvidence) {
					if(ArrayIsInList(correctEvidence, foundSent)) {
						correctSentences.add(foundSent);
						countCorrect++;
						if(!correct) {
							correct = true;
							countClaimCorrect++;
						}
					}
					else{
						countWrong++;
						wrongSentences.add(foundSent);
						if(!wrong) {
							wrong = true;
							countClaimWrong++;
						}
					}
				}
				boolean atLeastOne = false;
				
				if (!label.equals("NOT ENOUGH INFO")) {
					verifiableClaimCount++;
					for(Object[] correctSent : correctEvidence) {
						if(!ArrayIsInList(foundEvidence, correctSent)) {
							missedSentences.add(correctSent);
							countMissed++;
						}
						else{
							atLeastOne = true;
						}
					}
					if(!atLeastOne) {
						countClaimMissed++;
					}
				} 
				else if(label.equals("NOT ENOUGH INFO") && foundEvidence.isEmpty()) {
					countClaimCorrect++;
				}

				
				Map<String, Object> claimVals = new HashMap<String, Object>();

				claimVals.put("claim", claim);
				claimVals.put("label", label);
				claimVals.put("missed sentences", missedSentences);
				claimVals.put("correct sentences", correctSentences);
				claimVals.put("incorrect sentences", wrongSentences);
				
				writer.append(new JSONObject(claimVals).toString());
				writer.append("\n");

			}

			double totalPrecision = countCorrect / (countCorrect + countWrong);
			double totalRecall = countCorrect / (countCorrect + countMissed);
			writer.append("\n\n\n\n");
			writer.append("Number of claims with a correct sentence found (or it correctly found none): " + countClaimCorrect+"/"+claimCount+"\n");
			writer.append("Number of sentences where some of the correct sentences were not found: " + countMissed+"/"+claimCount+"\n");
			writer.append("Number of verifiable sentences where none of the correct sentences were found: " + countClaimMissed+"/"+verifiableClaimCount+"\n");
			writer.append("Number of incorrect sentences found: " + countWrong+"\n");
			writer.append("Number of claims with incorrect sentences found: " + countClaimWrong+"/"+claimCount+"\n");
			writer.append("Number of verifiable claims " + verifiableClaimCount + "\n");
            writer.append("Precision: " + totalPrecision + ", Recall: " + totalRecall + ", f1: " + f1(totalPrecision, totalRecall) + "\n");

            System.out.println("Precision: " + totalPrecision + ", Recall: " + totalRecall + ", f1: " + f1(totalPrecision, totalRecall) + "\n");

			answersReader.close();
			resultsReader.close();
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (JSONException e) {
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

	 private static double f1(double precision, double recall) {
	    if (precision + recall == 0) {
	        return 0;
        }
	    return 2 * precision * recall / (precision + recall);

     }

}

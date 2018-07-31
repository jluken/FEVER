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

public class DocFinderTester {
	static String answersFileName = "shared_task_dev.jsonl";
	static String resultsFileName = "found_documents.jsonl";
	static String analysisFileName = "document_analysis.jsonl";
	
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
			double countPrimaryCorrect = 0;
			double countWrong = 0;
			double countPrimaryWrong = 0;
			double countMissed = 0;
			double countPrimaryMissed = 0;
			int claimCount = 0;
			int verifiableClaimCount = 0;

			while(resultsReader.hasNext()) {
				claimCount++;
				String answer = Normalizer.normalize(answersReader.nextLine(), Normalizer.Form.NFC);	
				JSONObject answerJson = new JSONObject(answer);
				JSONArray answerEvidence = (JSONArray) answerJson.get("evidence");
				String label = answerJson.getString("label");
				ArrayList<Object[]> correctEvidence = new ArrayList<Object[]>();
				for(int i = 0; i < answerEvidence.length(); i++) {
					JSONArray evidenceSet = answerEvidence.getJSONArray(i);
					JSONArray primarySentence = evidenceSet.getJSONArray(0);
					if(!primarySentence.get(2).equals(null) && evidenceSet.length() == 1) {
						String wikiName = primarySentence.get(2).toString();
						Integer sentNum = primarySentence.getInt(3);
						Object[] answerArr = {wikiName, sentNum};
						if(!DocIsInList(correctEvidence, answerArr)) {
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
						if(!DocIsInList(foundEvidence, answerArr)) {
							foundEvidence.add(answerArr);
						}
					}
				}

				ArrayList<Object[]> correctDocs = new ArrayList<Object[]>();
				ArrayList<Object[]> wrongDocs = new ArrayList<Object[]>();
				ArrayList<Object[]> missedDocs = new ArrayList<Object[]>();

				for(Object[] foundSent : foundEvidence) {
					if(DocIsInList(correctEvidence, foundSent)) {
						correctDocs.add(foundSent);
						countCorrect++;
						if((int)foundSent[1] == 0) {
							countPrimaryCorrect++;
						}
					}
					else{
						countWrong++;
						wrongDocs.add(foundSent);
						if((int)foundSent[1] == 0) {
							countPrimaryWrong++;
						}
					}
				}
				boolean missed = false;
				if (!label.equals("NOT ENOUGH INFO")) {
					verifiableClaimCount++;
					for(Object[] correctSent : correctEvidence) {
						if(!missed && !correctSent[0].equals("multi")) {
							if(!DocIsInList(foundEvidence, correctSent)) {
								missed = true;
								missedDocs.add(correctSent);
								countMissed++;
							}
							if(!DocIsInPrimaryList(foundEvidence, correctSent)) {
								missed = true;
								countPrimaryMissed++;
							}
						}
					}
				} 


				
				Map<String, Object> claimVals = new HashMap<String, Object>();

				claimVals.put("claim", claim);
				claimVals.put("label", label);
				claimVals.put("missed docs", missedDocs);
				claimVals.put("correct docs", correctDocs);
				claimVals.put("incorrect docs", wrongDocs);
				
				writer.append(new JSONObject(claimVals).toString());
				writer.append("\n");

			}

			double totalPrecision = countCorrect / (countCorrect + countWrong);
			double totalRecall = countCorrect / (countCorrect + countMissed);
			double primaryPrecision = countPrimaryCorrect / (countPrimaryCorrect + countPrimaryWrong);
			double primaryRecall = countCorrect / (countPrimaryCorrect + countPrimaryMissed);
			writer.append("\n\n\n\n");
			writer.append("Number of verifiable claims with a correct document found: " + countCorrect+"/"+verifiableClaimCount+"\n");
			writer.append("Number of verifiable claims with a correct document found in primary: " + countPrimaryCorrect+"/"+verifiableClaimCount+"\n");
			writer.append("Number of sentences where the correct document was not found: " + countMissed+"/"+claimCount+"\n");
			writer.append("Number of incorrect documents found: " + countWrong+"\n");
			writer.append("Number of incorrect documents found in primary: " + countPrimaryWrong+"\n");
            writer.append("Precision: " + totalPrecision + ", Recall: " + totalRecall + ", f1: " + f1(totalPrecision, totalRecall) + "\n");
            writer.append("Primary Precision: " + primaryPrecision + ", Primary Recall: " + primaryRecall + ", Primary f1: " + f1(primaryPrecision, primaryRecall) + "\n");

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
	
	private static boolean DocIsInList(ArrayList<Object[]> list, Object[] arr) {
		boolean inside = false;
		for(Object[] listArr : list) {
			if(listArr[0].equals(arr[0])) {
				inside = true;
			}
		}
		return inside;
	}
	
	private static boolean DocIsInPrimaryList(ArrayList<Object[]> list, Object[] arr) {
		boolean inside = false;
		for(Object[] listArr : list) {
			if(listArr[0].equals(arr[0]) && (int)listArr[1] == 0) {
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
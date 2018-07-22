import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Scanner;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class FEVERTester {
	static String answersFileName = "shared_task_dev.jsonl";
	static String resultsFileName = "dev_predicted_evidence.jsonl";
	static String analysisFileName = "FEVER_analysis.jsonl";
	
	public static void main(String[] args) {
		try {
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

				JSONArray wikiLines = (JSONArray) resultJson.get("sentences");
				ArrayList<Object[]> foundEvidence = new ArrayList<Object[]>();
				for(int i = 0; i < resultEvidence.length(); i++) {
					JSONArray resultSet = resultEvidence.getJSONArray(i);
					JSONArray primarySentence = resultSet.getJSONArray(0);
					if(!primarySentence.get(2).equals(null)) {
						String wikiName = primarySentence.get(2).toString();
						Integer sentNum = primarySentence.getInt(3);
						Object[] answerArr = {wikiName, sentNum};
						if(!ArrayIsInList(correctEvidence, answerArr)) {
							correctEvidence.add(answerArr);
						}
					}
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

                if (!label.equals("NOT ENOUGH INFO")) {
                    verifiableClaimCount++;
                    int foundEvidenceCorrectCount = correctSentences.size();
                    // precision = # correct evidence found / # correct evidence sets in total
                    precision = (double)foundEvidenceCorrectCount / goldEvidenceCount;
                    // recall = # correct evidence found / # evidence found in total
                    if (foundEvidence.size() == 0) {
                        recall = 0;
                        verifiableClaimWithNoEvidenceFound ++;
                        System.out.println("No evidence found for:");
                        System.out.println(claim);
                    }
                    else {
                        recall = (double) foundEvidenceCorrectCount / foundEvidence.size();
                    }
                }

                precisionSum += precision;
				recallSum += recall;

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
					try {
					    writer.append(wikiLines.getString((int) answerSent[1]) + "\n");
                    }
                    catch (JSONException e) {
						System.out.println("Wiki page unmatched");
					    System.out.println(claim);
					    System.out.println(wikiLines);
					    System.out.println(answerEvidence);
                    }
				}
				writer.append("\nCorrectly found sentences: \n");
				for(Object[] resultSent : correctSentences) {
					writer.append(wikiLines.getString((int) resultSent[1]) + "\n");
				}
				writer.append("\nIncorrectly found sentences: \n");
				for(Object[] resultSent : wrongSentences) {
					writer.append(wikiLines.getString((int) resultSent[1]) + "\n");
				}
				writer.append("\nPrecision: " + precision + ", Recall: " + recall + ", f1: " + f1(precision, recall) + "\n");

			}

			double totalPrecision = precisionSum / verifiableClaimCount;
			double totalRecall = recallSum / verifiableClaimCount;
			writer.append("\n\n\n\n");
			writer.append("Number of claims with a correct sentence found (or it correctly found none): " + countCorrect+"/"+claimCount+"\n");
			writer.append("Number of sentences where none of the correct sentences were found: " + countMissed+"/"+claimCount+"\n");
			writer.append("Number of incorrect sentences found: " + countWrong+"\n");
			writer.append("Number of claims with incorrect sentences found: " + countSentenceWrong+"/"+claimCount+"\n");
			writer.append("Number of verifiable claims " + verifiableClaimCount + "\n");
            writer.append("Precision: " + totalPrecision + ", Recall: " + totalRecall + ", f1: " + f1(totalPrecision, totalRecall) + "\n");

            System.out.println("Precision: " + totalPrecision + ", Recall: " + totalRecall + ", f1: " + f1(totalPrecision, totalRecall) + "\n");

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

	 private static double f1(double precision, double recall) {
	    if (precision + recall == 0) {
	        return 0;
        }
	    return 2 * precision * recall / (precision + recall);

     }

}

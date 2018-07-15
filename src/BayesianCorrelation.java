import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.Normalizer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Tree;

public class BayesianCorrelation {
	static String claimsFileName = "train.jsonl";
	static String correlationsResultsFileName = "rootCorrelations.txt";
	static String wikiDirName = "wiki-dump";
	static String rootCorrelations = "rootCorrelations.ser";
	static int numClaimsToTest = 150000;
	
	static Map<String, Map<String, Object>> wikiMap;
	//static Map<String, ArrayList<String>> disambiguationMap;
	
	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    System.out.println("Beginning document processing. Time: "+dtf.format(LocalDateTime.now()));	    
		
		String statement = "";
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma");
	    props.setProperty("coref.algorithm", "neural");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(LocalDateTime.now()));
		try {
			Scanner claimReader = new Scanner(new FileReader(claimsFileName));
			File oldFile = new File(correlationsResultsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(correlationsResultsFileName, true));
			try {
		         FileInputStream fileIn = new FileInputStream("wikiMap.ser");
		         ObjectInputStream in = new ObjectInputStream(fileIn);
		         System.out.println("Attempting to read wikiMap object");
		         wikiMap = (Map<String, Map<String, Object>>) in.readObject();
		         in.close();
		         fileIn.close();
		      } catch (IOException i) {
		         System.out.println("Unable to read wikiMap object. Generating new object.");
		         getWikiMap(wikiDirName);
				 try {
			         FileOutputStream fileOut = new FileOutputStream("wikiMap.ser");
			         ObjectOutputStream out = new ObjectOutputStream(fileOut);
			         out.writeObject(wikiMap);
			         out.close();
			         fileOut.close();
			      } catch (IOException io) {
			         io.printStackTrace();
			      }
		      } catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
			
			
			System.out.println("wikiMap compiled. Time: "+dtf.format(LocalDateTime.now()));
			int claimCount =0;
			Map<String, Map<String, Object>> rootCorrelation = new HashMap<String, Map<String, Object>>();
			while(claimReader.hasNext() && claimCount < numClaimsToTest) {	
				claimCount++;
				statement = claimReader.nextLine();
				try {
				    JSONObject claimJson = new JSONObject(statement);
				    String claim = Normalizer.normalize(claimJson.getString("claim"), Normalizer.Form.NFC);
				    String label = claimJson.getString("label");
				    JSONArray answerEvidence = (JSONArray) claimJson.get("evidence");
				    if(claimCount % 100 == 0) {
				    	System.out.println("Claim "+ claimCount + ". Time: "+dtf.format(LocalDateTime.now()));
				    }
				    String formattedClaim = formatSentence(claim);
				    formattedClaim = formattedClaim.substring(0, formattedClaim.length() - 1);
				    CoreDocument document = new CoreDocument(claim);
				    pipeline.annotate(document);
				    CoreSentence claimSen = document.sentences().get(0);
				    SemanticGraph dependencyGraph = claimSen.dependencyParse();
				    String root = dependencyGraph.getFirstRoot().originalText().toLowerCase();
				    //System.out.println("root: " + root);
				    
				    String[] sentences = {};
				    
				    ArrayList<Integer> evidenceSentences = new ArrayList<Integer>();
				    String wikiName = "";
				    for(int i = 0; i < answerEvidence.length(); i++) {
				    	JSONArray evidenceSet = answerEvidence.getJSONArray(i);
						JSONArray primarySentence = evidenceSet.getJSONArray(0);
						
						int sentNum;
						if(!primarySentence.get(2).equals(null) && evidenceSet.length() == 1) { //only count solo evidence?
							wikiName = Normalizer.normalize(primarySentence.get(2).toString(), Normalizer.Form.NFC);
							sentNum = primarySentence.getInt(3);
							if(!evidenceSentences.contains(sentNum)) {
								evidenceSentences.add(sentNum);
							}
						}
				    }
				    evidenceSentences = (ArrayList<Integer>) evidenceSentences.stream().distinct().collect(Collectors.toList());
				    //System.out.println("Solo evidence: " + evidenceSentences.toString());
	
					ArrayList<String> wikiArrayList= new ArrayList<String>();
					wikiArrayList.add(wikiName.toLowerCase());
					ArrayList<Map<String, String>> allDocs = getDocsFromTopics(wikiArrayList);
					if(allDocs.size() == 0) {
						continue;
					}
					Map<String, String> wikiDoc = allDocs.get(0);
					String lines = Normalizer.normalize(wikiDoc.get("lines"), Normalizer.Form.NFC);
					sentences = lines.split("\\\n\\d*\\\t");

					Map<String, Integer> evidenceWords = new HashMap<String, Integer>();
					Map<String, Integer> totalWords = new HashMap<String, Integer>();
					int sentCount = 0;
					for(int i = 0; i < sentences.length; i++) {
						if(sentences[i].isEmpty()) {
							continue;
						}
						String sentence = sentences[i].split("\\t")[0].toLowerCase();
						sentCount++;
						//System.out.println("analyzing sentence: " + sentences[i]);
						//CoreDocument evidenceDocument = new CoreDocument(sentences[i]);
					    //pipeline.annotate(evidenceDocument);
					    //CoreSentence evidenceDoc = evidenceDocument.sentences().get(0);
						//Tree constituencyTree = evidenceDoc.constituencyParse();
						ArrayList<String> words = getWords(wikiName, sentence);
						//ArrayList<String> nav = getNounsAdjectivesAndVerbs(wikiName, constituencyTree);
	
						//System.out.println("nouns, adjectives, and verbs: " + nav.toString());
						
						for(String word : words) {
							if(totalWords.containsKey(word)) {
								int newNum = totalWords.get(word) + 1;
								totalWords.put(word, newNum);
							}
							else {
								totalWords.put(word, 1);
							}
							
							if(evidenceSentences.contains(i)) {
								if(evidenceWords.containsKey(word)) {
									int newNum = evidenceWords.get(word) + 1;
									evidenceWords.put(word, newNum);
								}
								else {
									evidenceWords.put(word, 1);
								}
							}
						}
		
				    }
	
				    if(rootCorrelation.containsKey(root)) {
				    	int existingNum = (Integer) rootCorrelation.get(root).get("sentCount");
				    	int existingEvNum = (Integer) rootCorrelation.get(root).get("evSentCount");
				    	Map<String, Integer> existingWords = (Map<String, Integer>) rootCorrelation.get(root).get("words");
				    	Map<String, Integer> existingEvWords = (Map<String, Integer>) rootCorrelation.get(root).get("evWords");
				    	for(String word : evidenceWords.keySet()) {
				    		if(existingEvWords.containsKey(word)) {
				    			int oldVal = existingEvWords.get(word);
				    			int newNum = oldVal + evidenceWords.get(word);
				    			existingEvWords.put(word, newNum);
				    		} else {
				    			existingEvWords.put(word, evidenceWords.get(word));
				    		}
				    	}
				    	for(String word : totalWords.keySet()) {
				    		if(existingWords.containsKey(word)) {
				    			int oldVal = existingWords.get(word);
				    			int newNum = oldVal + totalWords.get(word);
				    			existingWords.put(word, newNum);
				    		} else {
				    			existingWords.put(word, totalWords.get(word));
				    		}
				    	}
				    	Map<String, Object> rootMap = new HashMap<String, Object>();
				    	rootMap.put("evSentCount", existingEvNum + evidenceSentences.size());
				    	rootMap.put("sentCount", existingNum + sentCount);
				    	rootMap.put("evWords", existingEvWords);
				    	rootMap.put("words", existingWords);
				    	rootCorrelation.put(root, rootMap);
				    } else {
				    	Map<String, Object> rootMap = new HashMap<String, Object>();
				    	rootMap.put("evSentCount", evidenceSentences.size());
				    	rootMap.put("sentCount", sentCount);
				    	rootMap.put("evWords", evidenceWords);
				    	rootMap.put("words", totalWords);
				    	rootCorrelation.put(root, rootMap);
				    }
				}catch(Exception e) {
					System.out.println("error in claim "+ claimCount);
					e.printStackTrace();
				}
			//end of claim
			}
			System.out.println("Finished initial mapping. Time: " + dtf.format(LocalDateTime.now()));
			
			Map<String, Map<String, Float>> topWords = new HashMap<String, Map<String, Float>>();  
		    for(String rootWord : rootCorrelation.keySet()) {
		    	Map<String, Float> wordProbs = new HashMap<String, Float>();
		    	float totalSent = ((Integer) rootCorrelation.get(rootWord).get("sentCount")).floatValue();
		    	float evSent = ((Integer) rootCorrelation.get(rootWord).get("evSentCount")).floatValue();

		    	for(String corrWord : ((Map<String, Integer>) rootCorrelation.get(rootWord).get("evWords")).keySet()) {
		    		int evWordCount = ((Map<String, Integer>) rootCorrelation.get(rootWord).get("evWords")).get(corrWord);
		    		int totalWordCount = ((Map<String, Integer>) rootCorrelation.get(rootWord).get("words")).get(corrWord);
			    	float prob = evWordCount/((float) totalWordCount);
			    	if(evWordCount >= 5 && prob > 0.7) {
			    		wordProbs.put(corrWord, prob);
		    		}
		    	}
				if(!wordProbs.isEmpty()) {
					topWords.put(rootWord, wordProbs);
					writer.append(rootWord + ": [" + wordProbs.toString() + "]");
			    	writer.append("\n");
				}

		    	
		    }
		    System.out.println("Finished creating top map. Time: " + dtf.format(LocalDateTime.now()));
		    //System.out.println("keyset: " + topWords.keySet().toString());
		    
		    try {
		        FileOutputStream fileOut = new FileOutputStream(rootCorrelations);
		        ObjectOutputStream out = new ObjectOutputStream(fileOut);
		        out.writeObject(topWords);
		        out.close();
		        fileOut.close();
		     } catch (IOException io) {
		        io.printStackTrace();
		     }
			claimReader.close();
			writer.close();
		}catch (FileNotFoundException e) {
			System.out.println("Could not open file  "+claimsFileName);
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("There was an IO Exception during statement " + statement);
			e.printStackTrace();
		}
		
	}
	
	public static void getWikiMap(String wikiDirName) {
		wikiMap = new HashMap<String, Map <String, Object>>();
		//disambiguationMap = new HashMap<String, ArrayList<String>>();
		
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
							String wikiEntry = s.nextLine();
						    JSONObject wikiJson = new JSONObject(wikiEntry);
						    String id = Normalizer.normalize(wikiJson.getString("id"), Normalizer.Form.NFC).toLowerCase();
						    if(!id.isEmpty()) {
						    	String fileName = wikiEntryList.getName();
						    	Map<String, Object> docLocation = new HashMap<String, Object>();
						    	docLocation.put("fileName", fileName);
						    	docLocation.put("offset", byteOffset);
								wikiMap.put(id, docLocation);
								
//								int paren = id.indexOf("-lrb-");
//								if(paren > 0) {
//									String base = id.substring(0, paren-1).toLowerCase();
//									ArrayList<String> disambiguationChildren = new ArrayList<String>();
//									if(disambiguationMap.containsKey(base)) {
//										disambiguationChildren = disambiguationMap.get(base);
//										disambiguationChildren.add(id);
//										disambiguationMap.put(base, disambiguationChildren);
//									}
//									else {
//										disambiguationChildren.add(id);
//										disambiguationMap.put(base, disambiguationChildren);
//									}
//								}
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
	
	private static String formatSentence(String sentence) {
		String newSent = sentence.toLowerCase();
		newSent = newSent.replaceAll(",", " ,").replaceAll("\\.", " .").replaceAll(";", " ;").replaceAll(":", " :").replaceAll("\'s", " \'s");
		newSent = newSent.replaceAll("-lrb-", "-lrb- ").replaceAll("-rrb-", " -rrb-").replaceAll("-rsb-", " -rsb-").replaceAll("-lsb-", " -lsb-");
		newSent = newSent.replaceAll("\\(", "-lrb- ").replaceAll("\\)", " -rrb-").replaceAll("\\]", " -rsb-").replaceAll("\\[", " -lsb-");
		return newSent;
	}
	
	private static ArrayList<Map<String, String>> getDocsFromTopics(ArrayList<String> possibleTopics) {
		ArrayList<Map<String, String>> wikiDocs = new ArrayList<Map<String, String>>();
		for(String topic: possibleTopics) {
			String urlTitle = topic.replace(' ', '_').toLowerCase();
			Map<String, String> wikiDoc = new HashMap<String, String>();
//			boolean emptyDisam = false;
			if (!topic.isEmpty() && wikiMap.containsKey(urlTitle)){
				Map<String, Object> fileInfo = wikiMap.get(urlTitle);
				String wikiListName = wikiDirName + "\\" + fileInfo.get("fileName");
				try {
					BufferedReader reader = new BufferedReader(new FileReader(wikiListName)); 
					Long byteOffset = (Long) fileInfo.get("offset");
					reader.skip(byteOffset);
				    String wikiEntry = reader.readLine();
				    JSONObject wikiJson = new JSONObject(wikiEntry);
//				    if (wikiJson.getString("text").toLowerCase().contains((topic+" may refer to : "))) {
//			    		ArrayList<Map<String, String>> disambiguationChildren = findDisambiguationChildren(wikiJson);
//			    		wikiDocs.addAll(disambiguationChildren);
//			    		if(disambiguationChildren.isEmpty()) {
//			    			emptyDisam = true;
//			    		}
//			    	}
//				    else {
//					    wikiDoc.put("id", wikiJson.getString("id"));
//					    wikiDoc.put("text", wikiJson.getString("text"));
//					    wikiDoc.put("lines", wikiJson.getString("lines"));
//					    wikiDocs.add(wikiDoc);
//				    }
				    wikiDoc.put("id", (Normalizer.normalize(wikiJson.getString("id"), Normalizer.Form.NFC)));
				    wikiDoc.put("text", (Normalizer.normalize(wikiJson.getString("text"), Normalizer.Form.NFC)));
				    wikiDoc.put("lines", (Normalizer.normalize(wikiJson.getString("lines"), Normalizer.Form.NFC)));
				    wikiDocs.add(wikiDoc);
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
			
//			if (!topic.isEmpty() && disambiguationMap.containsKey(urlTitle)){
//				//System.out.println("disambiguation found: " + urlTitle+". Time: "+dtf.format(LocalDateTime.now()));
//				if(emptyDisam) {
//					ArrayList<String> disambiguations = disambiguationMap.get(urlTitle);
//					ArrayList<String> unchecked = new ArrayList<String>(disambiguations);
//					for(Map<String, String> wiki: wikiDocs) {
//						if(disambiguations.contains(wiki.get("id").toLowerCase())) {
//							unchecked.remove(wiki.get("id"));
//						}
//					}
//					wikiDocs.addAll(getDocsFromTopics(unchecked));
//				}
//			}
			//System.out.println("Topic " + topic + " done. Time: " + dtf.format(LocalDateTime.now()));
		}
		return wikiDocs;
	}
	


private static ArrayList<String> getWords(String wikiName, String claim) {
	String wikiTitle = formatSentence(wikiName.replaceAll("_", " "));
	String[] words = claim.split(" |\\t");
	

    //return ArrayList of topic phrases, from broadest to most narrow
    ArrayList<String> wordList = new ArrayList<String>();
    //String[] navTags = {"NN", "NNS", "NNP", "NNPS", "NP", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};
    String[] skipWords = {"a", "an", "the", "at", "by", "down", "for", "from", "in", "into", "like", "near", "of", "off", "on", "onto", "onto", "over", 
			"past", "to", "upon", "with", "and", "&", "as", "but", "for", "if", "nor", "once", "or", "so", "than", "that", "till", "when", "yet", "'s", "'",
			"be", "is", "am", "are", "was", "were", "been", "being", "has", "have", "had", "having", "do", "does", "did",
			"he", "his", "him", "she", "her", "hers", "it", "its", "they", "theirs",
			".","!","?",",",";",":", "-rrb-", "-lrb-", "-rsb-", "-lsb-", "", "0", "``", "''", "--", "-"};
    
    for(int i = 0; i < words.length; i++) {
    	String word = words[i];
    	if(word.length() > 0 && Character.isDigit(word.charAt(0))){
    		word = "#";
    	}
    	if(!Arrays.asList(skipWords).contains(word) && !wikiTitle.contains(word)) {
    		wordList.add(word);
    	} 
    }

    wordList = (ArrayList<String>) wordList.stream().distinct().collect(Collectors.toList());
    return wordList;
}

}

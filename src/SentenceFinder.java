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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;


public class SentenceFinder {
	
	static String claimsFileName = "shared_task_dev.jsonl";
	static String sentenceResultsFileName = "sentence_results.jsonl";
	static String wikiDirName = "wiki-dump";
	static int numClaimsToTest = 100;
	
	static Map<String, Map<String, Object>> wikiMap;
	static Map<String, Map<String, Float>> correlationMap;
	//static Map<String, ArrayList<String>> disambiguationMap;

	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    System.out.println("Beginning document processing. Time: "+dtf.format(LocalDateTime.now()));	    
		
		StanfordCoreNLP pipeline = establishPipeline();
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(LocalDateTime.now()));	    
	    compileWikiMaps();
		System.out.println("wikiMap compiled. Time: "+dtf.format(LocalDateTime.now()));
		compileCorrelationMap();
		System.out.println("correlationMap compiled. Time: "+dtf.format(LocalDateTime.now()));
		
		int claimCount =0;
		try {
			Scanner claimReader = new Scanner(new FileReader(claimsFileName));
			File oldFile = new File(sentenceResultsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(sentenceResultsFileName, true));
			
			
			while(claimReader.hasNext() && claimCount < numClaimsToTest) {
				claimCount++;
				System.out.println("Finding sentences for claim "+ claimCount + ". Time: " + dtf.format(LocalDateTime.now()));
				try {
					String claimInfo = claimReader.nextLine();	
				    String sentenceResults = findSentencesFromWiki(pipeline, claimInfo);		    
				    writer.append(sentenceResults);
				    writer.append("\n");
				}catch(Exception e){
					e.printStackTrace();
					System.out.println("Something went wrong with processeing claim " + claimCount + ". returning null and skipping");
					String[] wikiSentences = {};
					ArrayList<Object[]> evidenceSets = new ArrayList<Object[]>();
					writer.append(sentenceResults(0, "", "", wikiSentences, evidenceSets));
				    writer.append("\n");
				}
			    
			}
			claimReader.close();
			writer.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
	}



	private static String findSentencesFromWiki(StanfordCoreNLP pipeline, String claimInfo) throws JSONException, IOException {
		JSONObject claimJson = new JSONObject(claimInfo);
		String claim = Normalizer.normalize(claimJson.getString("claim"), Normalizer.Form.NFC);   
		String label = claimJson.getString("label");
		int id = claimJson.getInt("id");
		JSONArray answerEvidence = (JSONArray) claimJson.get("evidence");
		System.out.println("Claim: "+claim);
		String wikiName = answerEvidence.getJSONArray(0).getJSONArray(0).get(2).toString();
		wikiName = Normalizer.normalize(wikiName, Normalizer.Form.NFC);
		if(wikiName.equals("null")) {
			String[] wikiSentences = {};
			ArrayList<Object[]> evidenceSets = new ArrayList<Object[]>();
			return sentenceResults(id, claim, label, wikiSentences, evidenceSets);
		}
		
		CoreDocument document = new CoreDocument(claim);
		pipeline.annotate(document);
		CoreSentence claimDoc = document.sentences().get(0);
		Tree constituencyTree = claimDoc.constituencyParse();
		SemanticGraph dependencyGraph = claimDoc.dependencyParse();
		String root = dependencyGraph.getFirstRoot().originalText().toLowerCase();
		String formattedClaim = formatSentence(claim);
		ArrayList<String[]> claimNE = getNamedEntities(formattedClaim, pipeline);
		
		ArrayList<Object[]> evidenceSets = new ArrayList<Object[]>();
		String[] wikiSentences = {};
		ArrayList<String> wikiArrayList= new ArrayList<String>();
		wikiArrayList.add(wikiName.toLowerCase());
		ArrayList<Map<String, String>> wikiDocs = getDocsFromTopics(wikiArrayList);
		if(wikiDocs.size() == 0 || !hasSoloEvidence(answerEvidence)) {
			return sentenceResults(id, claim, label, wikiSentences, evidenceSets);
		}
		List<String> wikiSentencesLines = Arrays.asList(wikiDocs.get(0).get("lines").split("\n"));
		wikiSentences = (String[]) wikiSentencesLines.stream()
                                                     .map(i -> getWikiSentencefromLine(i))
                                                     .collect(Collectors.toList())
                                                     .toArray(new String[wikiSentencesLines.size()]);

		
		String wikiTitle = formatSentence(wikiName.replace("_", " ")).toLowerCase();
		System.out.println("wikiTitle: " + wikiTitle);
		ArrayList<String[]> claimNANE = getNounsAndNamedEntities(wikiTitle, constituencyTree, claimNE);
		
		for(int j = 0; j < wikiSentences.length; j++) {
			String sentence = getSentenceTextFromWikiLines(wikiSentences[j]); //TODO: deal with duplicates
			if(containsNamedEntities(sentence, claim, claimNANE, wikiTitle, pipeline, root) ||
					 containsCorrelatedWord(sentence, root, wikiTitle) ||
					 containsValidRoot(sentence, root, pipeline)) {
				Object[] evidence = new Object[3];
				evidence[0] = wikiName;
				evidence[1] = sentence;
				evidence[2] = j;
				evidenceSets.add(evidence);
			}	
		}

		System.out.println();
			
		return sentenceResults(id, claim, label, wikiSentences, evidenceSets);
	}

	private static boolean hasSoloEvidence(JSONArray answerEvidence) {
		boolean isAlone = false;
		try {
			for(int i = 0; i < answerEvidence.length(); i++) {
				JSONArray evidenceSet;
			
				evidenceSet = answerEvidence.getJSONArray(i);
				if(evidenceSet.length() == 1) {
					isAlone = true;
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return isAlone;
	}


	private static String sentenceResults(int id, String claim, String label,
			String[] wikiSentences, ArrayList<Object[]> evidenceSets) throws JSONException {
		Map<String, Object> claimJsonMap = new LinkedHashMap<String, Object>();
		claimJsonMap.put("id", id);
		claimJsonMap.put("claim", claim);
		claimJsonMap.put("label", label);
		claimJsonMap.put("sentences", wikiSentences);
		claimJsonMap.put("evidence", evidenceSets.toArray(new Object[evidenceSets.size()]));
		return new JSONObject(claimJsonMap).toString();
	}



	private static String getSentenceTextFromWikiLines(String sentenceInfo) {
		String sentence;
		String[] tabs = sentenceInfo.split("\\t");
		if(tabs[0].equals("0")) {
			sentence = tabs[1];
		} else {
			sentence = tabs[0];
		}
		return sentence;
	}



	private static StanfordCoreNLP establishPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma, ner");
	    props.setProperty("coref.algorithm", "neural");
	    props.put("ner.model", "english.conll.4class.distsim.crf.ser.gz");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}
	
	private static void compileWikiMaps() {
		try {
	         FileInputStream fileIn = new FileInputStream("wikiMap.ser");
	         ObjectInputStream in = new ObjectInputStream(fileIn);
	         System.out.println("Attempting to read wikiMap object");
	         wikiMap = (Map<String, Map<String, Object>>) in.readObject();
	         in.close();
	         fileIn.close();
	    } catch (Exception e) {
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
		}
	}	
	
	private static void compileCorrelationMap() {
		try {
		     FileInputStream fileIn = new FileInputStream("rootCorrelations.ser");
		     ObjectInputStream in = new ObjectInputStream(fileIn);
		     correlationMap = (Map<String, Map<String, Float>>) in.readObject();
		     in.close();
		     fileIn.close();
		  } catch (Exception e) {
			  e.printStackTrace();
			  correlationMap = new HashMap<String, Map<String, Float>>();
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
							String wikiEntry = Normalizer.normalize(s.nextLine(), Normalizer.Form.NFD);
						    JSONObject wikiJson = new JSONObject(wikiEntry);
						    String id = wikiJson.getString("id").toLowerCase();
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
		String newSent = sentence.replace(",", " ,").replace(".", " .").replace(";", " ;").replace(":", " :").replace("'s", " 's").replace("' ", " ' ");
		newSent = newSent.replace("-LRB-", "-LRB- ").replace("-RRB-", " -RRB-").replace("-RSB-", " -RSB-").replace("-LSB-", " -LSB-");
		newSent = newSent.replace("(", "-LRB- ").replace(")", " -RRB-").replace("]", " -RSB-").replace("[", " -LSB-");
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
				String wikiListName = wikiDirName + "/" + fileInfo.get("fileName");
				try {
					BufferedReader reader = new BufferedReader(new FileReader(wikiListName)); 
					Long byteOffset = (Long) fileInfo.get("offset");
					reader.skip(byteOffset);
					String wikiEntry = Normalizer.normalize(reader.readLine(), Normalizer.Form.NFC);
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
				    wikiDoc.put("id", wikiJson.getString("id"));
				    wikiDoc.put("text", wikiJson.getString("text"));
				    wikiDoc.put("lines", wikiJson.getString("lines"));
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

	
	private static boolean isVerb(String word, StanfordCoreNLP pipeline) {
		CoreDocument document = new CoreDocument(word);
	    pipeline.annotate(document);
	    CoreSentence claimDoc = document.sentences().get(0);
		Tree constituencyTree = claimDoc.constituencyParse();
		List<Tree> leaves = constituencyTree.getLeaves();
		Tree wordTree = leaves.get(0);
		Tree type = wordTree.parent(constituencyTree);
		String[] verbTags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};
		boolean verb = false;
		if(Arrays.asList(verbTags).contains(type.label().toString())) {
			verb = true;
		} 
		return verb;
		
		
	}

	private static ArrayList<String> getNouns(String wikiTitle, Tree constituencyTree) {
		List<Tree> leaves = constituencyTree.getLeaves();
	    int wordNum = 0;

	    ArrayList<String> wordList = new ArrayList<String>();
	    String[] nounsTags = {"NN", "NNS", "NNP", "NNPS", "NP"};//, "CD", "JJ"}; 
	    while(wordNum < leaves.size()) {
	    	Tree word = leaves.get(wordNum);
	    	Tree type = word.parent(constituencyTree);
	    	String wordStr = word.toString().toLowerCase();
	    	if(Arrays.asList(nounsTags).contains(type.label().toString()) && !wikiTitle.contains(wordStr)) {
	    		wordList.add(wordStr);
	    	} 
	    	wordNum++;
	    }
    
	    return wordList;
	}

	private static ArrayList<String[]> getNounsAndNamedEntities(String wikiTitle,  Tree constituencyTree, List<String[]> namedEntities){
		ArrayList<String> nouns = getNouns(wikiTitle, constituencyTree);
	    List<String[]> filteredNamedEntities = namedEntities.stream().filter(arr -> !wikiTitle.contains(arr[0]))
				.collect(Collectors.toList());
	    System.out.println("claim named entities: ");
	    for(String[] ne : filteredNamedEntities) {
	    	System.out.print(ne[0] + "(" + ne[1] + "), ");
	    }
	    System.out.println();
			
	    ArrayList<String[]> nounsAndNamedEntities = new ArrayList<String[]>(filteredNamedEntities);
		for(String noun: nouns) {
			boolean contained = false;
			for(String[] namedEntity: filteredNamedEntities) {
				if(namedEntity[0].contains(noun)) {
					contained = true;
				}
			}
			if(!contained) {
				String[] nounEntry = {noun, "NOUN"};
				nounsAndNamedEntities.add(nounEntry);
			}
		}
		
		return nounsAndNamedEntities;
	}

	private static boolean containsNamedEntities(String evidenceSentence, String claim, ArrayList<String[]> claimNE, String wikiTitle, StanfordCoreNLP pipeline, String root) {
		boolean containsEntities = true;
		//unnecessary if statements are included here for the benefit of a descriptive text log
		if(evidenceSentence.isEmpty()) {
			containsEntities = false;
		}
		else if(claimNE.isEmpty()) {
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("contains no entities.");
			containsEntities = false;
		}
		
		String[] entityToBeSwapped = null;
		for(String[] ne: claimNE) {
			if(!evidenceSentence.toLowerCase().contains(ne[0]) && entityToBeSwapped == null) {
				entityToBeSwapped = ne.clone();
			}
			else if(!evidenceSentence.toLowerCase().contains(ne[0]) && containsEntities) {
				System.out.println("sentence: " + evidenceSentence);
				System.out.println("Missing both: \"" + ne[0] + "\" and \"" + entityToBeSwapped[0] + "\"");
				containsEntities = false;
			}
			
		}
		if(entityToBeSwapped == null) {
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("contains all entites");
		}
		if(containsEntities && entityToBeSwapped != null && entityToBeSwapped[1] == "NOUN") {
			System.out.println("sentence: " + evidenceSentence);
			if(isNounComplement(claim) && isNounComplement(evidenceSentence)) {
				System.out.println("The only remaining entity is a generic noun, and it's a 'is a' sentence.");
			} else {
				System.out.println("The only remaining entity is a generic noun, which can't be reliably replaced.");
				containsEntities = false;
			}
		}
		boolean missingRootVerb = false;
		if(isVerb(root, pipeline) && !evidenceSentence.contains(root)) {
			missingRootVerb = true;
		}
		if(containsEntities && entityToBeSwapped != null && entityToBeSwapped[1] != "NOUN" && missingRootVerb) {
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("The entity cannot be swapped because the root word is a verb which is missing.");
			containsEntities = false;
		}
		if(containsEntities && entityToBeSwapped != null && entityToBeSwapped[1] != "NOUN" && !missingRootVerb) {
			List<String[]> evidenceEntities = getNamedEntities(evidenceSentence, pipeline).stream().filter(arr -> !wikiTitle.contains(arr[0]))
					.collect(Collectors.toList());
			System.out.println("sentence: " + evidenceSentence);
			System.out.println("entity " + entityToBeSwapped[0] + "(" + entityToBeSwapped[1] + ") is missing. Attempting to find replacement");
			System.out.print("sentence named entities: ");
		    for(String[] ne : evidenceEntities) {
		    	System.out.print(ne[0] + "(" + ne[1] + "), ");
		    }
		    System.out.println();
			containsEntities = false;
			for(String[] ne: evidenceEntities) {
				if(!containsEntities && ne[1].equals(entityToBeSwapped[1])) {
					containsEntities = true;
					System.out.println("Swap successful: " + ne[0] + " for " + entityToBeSwapped[0]);
				}
			}
		}
		
		return containsEntities;
	}
	
	private static boolean isNounComplement(String sentence) {
		String[] NCTags = {" is a ", " is an " , " was a " , " was an "};
		boolean isNC = false;
		for(String tag : NCTags) {
			if(sentence.contains(tag)) {
				isNC = true;
			}
		}
		return isNC;
	}
	private static boolean containsCorrelatedWord(String sentence, String root, String wikiTitle) {
		ArrayList<String> words = getWords(wikiTitle, sentence.toLowerCase());
		boolean correlated = false;
		if(correlationMap.containsKey(root)) {
			Map<String, Float> correlations = correlationMap.get(root);	
			for(String word: words) {
				if(correlations.containsKey(word)) {
					System.out.println("Sentence: " + sentence);
					System.out.println("Added via correlation of root " + root + " to word " + word);
					correlated = true;
				}
			}
		}
		return correlated;
	}
	
	private static boolean containsValidRoot(String sentence, String root, StanfordCoreNLP pipeline) {
		String[] isWords = {"is", "was", "be", "are", "were"};
		String rootAlone = " " + root + " ";
		boolean validRoot = false;
		if(isVerb(root, pipeline) && !Arrays.asList(isWords).contains(root) && sentence.toLowerCase().contains(rootAlone)) {
			System.out.println("Sentence: " + sentence);
			System.out.println("Added via presence of root " + root);
			validRoot = true;
		}
		return validRoot;
	}
	
	private static ArrayList<String[]> getNamedEntities(String sentence, StanfordCoreNLP pipeline){

		ArrayList<CoreLabel> tokens = getSentenceTokens(sentence, pipeline);
		ArrayList<String[]> namedEntities = new ArrayList<String[]>();
		String neTag = "";
		String namedEntity = "";
		boolean neActive = false;
		for(int i = 0; i < tokens.size(); i++) {
			String ne = tokens.get(i).get(NamedEntityTagAnnotation.class);
			if(!neActive && !ne.equals("O")) {
				neActive = true;
				neTag = ne;
				namedEntity += tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && ne.equals(neTag)) {
				namedEntity += " " + tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && !ne.equals(neTag) && !ne.equals("O")) {
				String[] neInfo = {namedEntity.toLowerCase(), neTag};
				namedEntities.add(neInfo);		
				namedEntity = "";
				neTag = ne;
				namedEntity = tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && ne.equals("O")) {
				neActive = false;
				String[] neInfo = {namedEntity.toLowerCase(), neTag};
				namedEntities.add(neInfo);		
				namedEntity = "";
				neTag = "";
			}
		}
		if(neActive) {
			String[] neInfo = {namedEntity.toLowerCase(), neTag};
			namedEntities.add(neInfo);
		}
		return namedEntities;
   }

	private static ArrayList<CoreLabel> getSentenceTokens(String sentence, StanfordCoreNLP pipeline){
		Annotation document = new Annotation(sentence);
	    pipeline.annotate(document);
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    ArrayList<CoreLabel> tokens = new ArrayList<CoreLabel>();
	    if(sentences.isEmpty()) {
	    	return tokens;
	    }
	    CoreMap tokenSen = sentences.get(0);
	    
	    
	    for (CoreLabel token: tokenSen.get(TokensAnnotation.class)) {
	        tokens.add(token);
	      }
	    return tokens;
	}
	
	private static ArrayList<String> getWords(String wikiTitle, String claim) {
		String[] words = claim.split(" |\\t");
		
	    ArrayList<String> wordList = new ArrayList<String>();
	    String[] skipWords = {"a", "an", "the", "at", "by", "down", "for", "from", "in", "into", "like", "near", "of", "off", "on", "onto", "onto", "over", 
				"past", "to", "upon", "with", "and", "&", "as", "but", "for", "if", "nor", "once", "or", "so", "than", "that", "till", "when", "yet", "'s", "'",
				"be", "is", "am", "are", "was", "were", "been", "being", "has", "have", "had", "having", "do", "does", "did",
				"he", "his", "him", "she", "her", "hers", "it", "its", "they", "theirs",
				".","!","?",",",";",":", "-rrb-", "-lrb-", "-rsb-", "-lsb-", "", "0", "``", "''", "--", "-"};
	    
	    for(int i = 0; i < words.length; i++) {
	    	String word = words[i];
	    	if(!Arrays.asList(skipWords).contains(word) && !wikiTitle.contains(word)) {
	    		wordList.add(word);
	    	} 
	    }

	    wordList = (ArrayList<String>) wordList.stream().distinct().collect(Collectors.toList());
	    return wordList;
	}

    /**
     * get the wiki sentence from a tab-separated line in wiki dump
     * @param line
     * @return the sentence text from the line
     */
    private static String getWikiSentencefromLine(String line) {
        String[] tabs = line.split("\t");
        if (tabs.length < 2) {
            return "";
        }
        return tabs[1];
    }

}

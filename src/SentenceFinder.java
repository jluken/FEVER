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
import java.lang.reflect.Array;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.neural.Embedding;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Constituent;
import edu.stanford.nlp.trees.LabeledScoredConstituentFactory;
import edu.stanford.nlp.util.Pair;
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
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;



public class SentenceFinder {
	
	static String claimsFileName = "train_with_evidence_text_links.jsonl";
	static String sentenceResultsFileName = "extended_keywords_sentence_results.jsonl";
	static String wikiDirName = "wiki-dump";
	static int numClaimsToTest = 100;
	static String embeddingFileName = "embeddings/glove.6B.50d.txt";

	static Map<String, Map<String, Object>> wikiMap;
	//static Map<String, ArrayList<String>> disambiguationMap;

    /**
     * Split lines in wiki dump
     * @param input -- "lines" element in wiki
     * @return  a list of sentences in the wiki
     */
    private static String[] splitWikiLines(String input) {
        String[] lines = input.split("\n");
        String[] output = new String[lines.length];
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            String[] tabs = line.split("\t");
            try {
                output[i] = tabs[1];
            }
            catch (IndexOutOfBoundsException e) {
                output[i] = "";
            }
        }
        return output;
    }

	
	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    System.out.println("Beginning document processing. Time: "+dtf.format(LocalDateTime.now()));	    
		
		String statement = "";
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma, ner");
	    props.setProperty("coref.algorithm", "neural");
	    props.put("ner.model", "english.conll.4class.distsim.crf.ser.gz");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(LocalDateTime.now()));
		try {
			Scanner claimReader = new Scanner(new FileReader(claimsFileName));
			File oldFile = new File(sentenceResultsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(sentenceResultsFileName, true));
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
            VocabSimilarity similarity = new VocabSimilarity(embeddingFileName);
			while(claimReader.hasNext() && claimCount < numClaimsToTest) {
				claimCount++;
				statement = claimReader.nextLine();
			    JSONObject claimJson = new JSONObject(statement);
			    String claim = claimJson.getString("claim");
			    String label = claimJson.getString("label");
			    JSONArray answerEvidence = (JSONArray) claimJson.get("evidence");
			    
			    System.out.println("Claim "+ claimCount + ":  \""+claim+"\" Time: "+dtf.format(LocalDateTime.now()));
			    String formattedClaim = formatSentence(claim);
			    formattedClaim = formattedClaim.toLowerCase();
			    formattedClaim = formattedClaim.substring(0, formattedClaim.length() - 1);
			    
			    
			    String wikiName = "";
			    ArrayList<Object[]> evidenceSets = new ArrayList<Object[]>();
                ArrayList<Object[]> extendedEvidenceSets = new ArrayList<Object[]>();
			    String[] sentences = {};
			    // relevant words to search for
			    ArrayList<String> relevantWords = new ArrayList<String>();

			    // matched evidence sentences
                ArrayList<String> matchedWords = new ArrayList<String>();

                // indices of matched evidence sentences
                ArrayList<Integer> matchedWordsNum = new ArrayList<Integer>();

			    for(int i = 0; i < answerEvidence.length(); i++) {
                    JSONArray evidenceSet = answerEvidence.getJSONArray(i);
                    JSONArray primarySentence = evidenceSet.getJSONArray(0);
                    boolean isAlone = (evidenceSet.length() == 1);
                    String newWikiName = primarySentence.get(2).toString();
                    if (!isAlone || wikiName.equals(newWikiName)) {
                        continue;
                    }
                    wikiName = primarySentence.get(2).toString();
                    ArrayList<String> wikiArrayList = new ArrayList<String>();
                    wikiArrayList.add(wikiName.toLowerCase());
                    ArrayList<Map<String, String>> allDocs = getDocsFromTopics(wikiArrayList);
                    if (allDocs.size() == 0) {
                        continue;
                    }


				    // annotate claim
                    CoreDocument document = new CoreDocument(claim);
                    pipeline.annotate(document);
                    CoreSentence claimDoc = document.sentences().get(0);

                    System.out.println("ready to grab. Time: "+dtf.format(LocalDateTime.now()));
                    // for each wikidoc in all found docs
					for (Map<String, String> wikiDoc : allDocs) {
    //					relevantWords = getRelevantWords(formattedClaim, wikiName);
                        relevantWords = getNouns(formattedClaim, wikiName, claimDoc.constituencyParse());
    //				    relevantWords = getNamedEntities(formattedClaim, wikiName, pipeline);

//                        System.out.println("relevantWords: " + relevantWords.toString() + " Time: "+dtf.format(LocalDateTime.now()));
                        String lines = wikiDoc.get("lines");
                        //sentences = lines.split("\\\n\\d*\\\t");
                        sentences = splitWikiLines(lines);

                        Set<String> relevantWordsSet = relevantWords.stream().collect(Collectors.toSet());


                        Pair<List<String>, List<Integer>> matchedResult = searchSentencesWithKeywords(sentences, relevantWordsSet);
                        matchedWords = (ArrayList<String>) matchedResult.first;
                        matchedWordsNum = (ArrayList<Integer>) matchedResult.second;
                        for(int j = 0; j < matchedWords.size(); j++) {
                            Object[] evidence = new Object[3];
                            evidence[0] = wikiName;
                            evidence[1] = matchedWords.get(j);
                            evidence[2] = matchedWordsNum.get(j);
                            evidenceSets.add(evidence);
                        }

                        System.out.println("Original keywords");
                        System.out.println(relevantWords);
                        System.out.println("Evidence matched by relevant words:");
                        System.out.println(matchedWords);

                        // search for 10 closest words of each relevantWords
                        // and search for the 10 closest words of the root of the evidence sentneces
                        Set<String> roots = getDependencyRoots(claim, claimDoc.dependencyParse());
                        relevantWords.addAll(roots);

                        // the set of new keywords to match
                        HashSet<String> closestRelevantWords = new HashSet<String>();
                        for (String word: relevantWords) {
                            Map<String, Double> neighborsWordMap = similarity.getClosestWordVectors(word, 10);
                            Set<String> neighborWords = neighborsWordMap.keySet();
                            closestRelevantWords.addAll(neighborWords);
                        }

                        // search with the new set of keywords
                        // if there is one keyword that matches, count the claim sentence as an evidence
                        Pair<List<String>, List<Integer>> extendedMatchedResult = searchSentencesWithKeywords(sentences, closestRelevantWords);
                        List<String> evidenceMatchedByClosestWords = extendedMatchedResult.first;
                        List<Integer> evidenceIndexMatchedByClosestWords = extendedMatchedResult.second;

                        System.out.println("Extended keywords");
                        System.out.println(closestRelevantWords);
                        System.out.println("Evidence matched by closest words:");
                        System.out.println(evidenceMatchedByClosestWords);

                    }

			    }

			    Map<String, Object> claimJsonMap = new LinkedHashMap<String, Object>();
			    claimJsonMap.put("id", claimJson.getInt("id"));
			    claimJsonMap.put("claim", claim);
			    claimJsonMap.put("label", label);
//			    claimJsonMap.put("sentences", sentences);
			    claimJsonMap.put("relevantWords", relevantWords.toArray(new Object[relevantWords.size()]));
			    claimJsonMap.put("predicted_evidence", evidenceSets.toArray(new Object[evidenceSets.size()]));
                claimJsonMap.put("predicted_evidence2", extendedEvidenceSets.toArray(new Object[extendedEvidenceSets.size()]));
                claimJsonMap.put("correct_evidence", claimJson.getJSONArray("evidence"));
			    JSONObject resultJson = new JSONObject(claimJsonMap);
			    writer.append(resultJson.toString());
			    writer.append("\n");
			//end of claim
			}
			claimReader.close();
			writer.close();
		}catch (FileNotFoundException e) {
			System.out.println("Could not open file  "+claimsFileName);
			e.printStackTrace();
		} catch (JSONException e) {
			System.out.println("There was a problem with the json from " + statement);
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
				String wikiListName = wikiDirName + "/" + fileInfo.get("fileName");
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
	
	private static ArrayList<String> getRelevantWords(String formattedClaim, String wikiName){
		ArrayList<String> relevantWords = new ArrayList<String>();
		ArrayList<String> ignoredWords =  new ArrayList<String>();
		String[] punctuation = {".","!","?",",",";",":", "-rrb-", "-lrb-", "-rsb-", "-lsb-"};
		String[] negation = {"not", "never", "isn't", "wasn't", "won't", "aren't", "weren't", "hasn't", "haven't", "hadn't", "don't", "doesn't"};
		String[] articles = {"a", "an", "the"};
		String[] ignoredVerbs = {"is", "was", "will", "are", "were", "has", "have", "had", "do", "does", "did", "there"};
		String[] ignoredPreps = {"for", "from", "of", "on", "in", "by", "'s"};
		String[] pronouns = {"he", "his", "him", "she", "her", "hers", "it", "its", "they", "theirs"};
		String[] conjunctions = {"and", "or", "nor", "but", "so"};
		String[] wikiTitle = formatSentence(wikiName.replaceAll("_", " ")).split(" ");
		ignoredWords.addAll(Arrays.asList(punctuation));
		ignoredWords.addAll(Arrays.asList(negation));
		ignoredWords.addAll(Arrays.asList(articles));
		ignoredWords.addAll(Arrays.asList(ignoredVerbs));
		ignoredWords.addAll(Arrays.asList(ignoredPreps));
		ignoredWords.addAll(Arrays.asList(pronouns));
		ignoredWords.addAll(Arrays.asList(conjunctions));
		ignoredWords.addAll(Arrays.asList(wikiTitle));
		
		
		String[] claimWords = formattedClaim.split(" ");
		for(String word: claimWords) {
			if(!ignoredWords.contains(word)) {
				relevantWords.add(word);
			}
		}
		return relevantWords;

	}
	
	
	private static ArrayList<String> getMaxNounPhrases(String claim, String wikiName, Tree constituencyTree) {
		
		ArrayList<String> nounPhrases = new ArrayList<String>();
		List<Tree> leaves = constituencyTree.getLeaves();
	    int phraseStart = 0;

	    //return ArrayList of topic phrases, from broadest to most narrow
	    ArrayList<Tree> nounTreeList = new ArrayList<Tree>();
	    String[] nounsAndPhrases = {"NN", "NNS", "NNP", "NNPS", "NP"};
	    while(phraseStart < leaves.size()) {
	    	Tree nounTree = leaves.get(phraseStart);
	    	Tree nounPhrase = nounTree.deepCopy();
	    	boolean noun = false;
	    	while(nounTree.parent(constituencyTree) != null) {
		    	nounTree = nounTree.parent(constituencyTree);
		    	if(Arrays.asList(nounsAndPhrases).contains(nounTree.label().toString())) {
		    		nounPhrase = nounTree.deepCopy();
		    		noun = true;
		    	}
		    }
	    	phraseStart += nounPhrase.getLeaves().size();
	    	if(noun) {
	    		nounTreeList.add(nounPhrase);
	    	}
	    	
	    }

	    //System.out.println("noun phrases 3");
	    for(Tree nounTree: nounTreeList) {
	    	List<Tree> nounWords = nounTree.getLeaves();
	    	String[] articles = {"a", "an", "the"};
	    	String nounPhrase;
	    	int start = 0;
	    	if(!Arrays.asList(articles).contains(nounWords.get(0).toString().toLowerCase())) {
	    		nounPhrase  = nounWords.get(0).toString().toLowerCase();	
	    		start = 1;
	    	}
	    	else {
	    		nounPhrase  = nounWords.get(1).toString().toLowerCase();	
	    		start = 2;
	    	}
		       
		    for(int j = start; j < nounWords.size(); j++) {
		    	boolean endsInPossessive = (j == nounWords.size()-1) && nounWords.get(j).toString().equals("\'s");
		    	if(!endsInPossessive) {
		    		nounPhrase += " ";
		    		nounPhrase += nounWords.get(j).toString().toLowerCase();
		    	}
		    }
		    String wikiTitle = formatSentence(wikiName.replaceAll("_", " "));
		    if(!nounPhrase.contains(wikiTitle)) {
		    	nounPhrases.add(nounPhrase.toLowerCase());
		    }    
	    }

	    
	    return nounPhrases;
	}
	
	private static ArrayList<String> getNouns(String claim, String wikiName, Tree constituencyTree) {
		String wikiTitle = formatSentence(wikiName.replaceAll("_", " "));
		List<Tree> leaves = constituencyTree.getLeaves();
	    int wordNum = 0;

	    //return ArrayList of topic phrases, from broadest to most narrow
	    ArrayList<String> wordList = new ArrayList<String>();
	    String[] nounsTags = {"NN", "NNS", "NNP", "NNPS", "NP"};//, "CD", "JJ"}
	    while(wordNum < leaves.size()) {
	    	Tree word = leaves.get(wordNum);
	    	Tree type = word.parent(constituencyTree);
	    	String wordStr = word.toString().toLowerCase();
	    	//System.out.println("word label: " + word.label().toString());
	    	//System.out.println("type label: " + type.label().toString());
	    	if(Arrays.asList(nounsTags).contains(type.label().toString()) && !wikiTitle.contains(wordStr)) {
	    		wordList.add(wordStr);
	    	} 
	    	wordNum++;
	    }
    
	    return wordList;
	}


    public static Set<String> getDependencyRoots(String claim, SemanticGraph dependencyGraph) {
        Collection<IndexedWord> roots = dependencyGraph.getRoots();
        Set<String> verbs = roots.stream().map(indexedWord -> indexedWord.word()).collect(Collectors.toSet());
        return verbs;
    }


	private static ArrayList<String> getNamedEntities(String sentence, String wikiName, StanfordCoreNLP pipeline){
		String wikiTitle = formatSentence(wikiName).toLowerCase();
		ArrayList<CoreLabel> tokens = getSentenceTokens(sentence, pipeline);
		ArrayList<String> namedEntities = new ArrayList<String>();
		String namedEntity = "";
		boolean neActive = false;
		for(int i = 0; i < tokens.size(); i++) {
			String ne = tokens.get(i).get(NamedEntityTagAnnotation.class);
			if(!neActive && !ne.equals("O")) {
				neActive = true;
				namedEntity += tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && !ne.equals("O")) {
				namedEntity += " " + tokens.get(i).get(TextAnnotation.class);
			}
			else if(neActive && ne.equals("O")) {
				neActive = false;
				if(!wikiTitle.contains(namedEntity.toLowerCase())){
					namedEntities.add(namedEntity.toLowerCase());
				}			
				namedEntity = "";
			}
		}
		if(neActive && !wikiTitle.contains(namedEntity.toLowerCase())) {
			namedEntities.add(namedEntity.toLowerCase());
		}
		return namedEntities;
   }

	private static ArrayList<CoreLabel> getSentenceTokens(String sentence, StanfordCoreNLP pipeline){
		Annotation document = new Annotation(sentence);
	    pipeline.annotate(document);
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    CoreMap tokenSen = sentences.get(0);
	    
	    ArrayList<CoreLabel> tokens = new ArrayList<CoreLabel>();
	    for (CoreLabel token: tokenSen.get(TokensAnnotation.class)) {
	        tokens.add(token);
	      }
	    return tokens;
	}

    /**
     * Match a set of keywords in a set of sentences
     * if at least one word in keywords appear in a sentence, the sentence is matched
     * @param sentences a list of sentences to match
     * @param keywords a list of keywords
     * @return a pair of matched sentences and the indices of those matched sentences
     */
	private static Pair<List<String>, List<Integer>> searchSentencesWithKeywords(String[] sentences,
                                                                                 Set<String> keywords) {

        ArrayList<String> matchedSentences = new ArrayList<>();
        ArrayList<Integer> matchedSentenceIndices = new ArrayList<>();
        for(int j = 0; j < sentences.length; j++) {
            String sentence = sentences[j].toLowerCase();
            boolean wordsPresent = false;
            for(String word: keywords) {
                if(sentence.contains(word)) {
                    wordsPresent = true;
                }
            }
            if(wordsPresent && keywords.size() > 0) {
                matchedSentences.add(sentence);
                matchedSentenceIndices.add(j);
            }
        }
        return new Pair(matchedSentences, matchedSentenceIndices);
    }

}

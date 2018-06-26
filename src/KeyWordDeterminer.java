import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class KeyWordDeterminer {
	
	static String sourceFileName = "train_with_evidence_text.jsonl";
	static String keywordsFileName = "keywords.jsonl";
	static int numClaimsToTest = 100;
	
	public static void main(String[] args) {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma, ner");
	    props.setProperty("coref.algorithm", "neural");
	    props.put("ner.model", "english.conll.4class.distsim.crf.ser.gz");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		try {
			//String testsen = "Fox 2000 Pictures released the film Soul Food.";
			//ArrayList<String> namedEntities = getNamedEntities(testsen, pipeline);
			//System.out.println(namedEntities.toString());
			
			Scanner trainTextReader = new Scanner(new FileReader(sourceFileName));
			File oldFile = new File(keywordsFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(keywordsFileName, true));
			
			int claimNum = 0;
		    while(trainTextReader.hasNext() && claimNum < numClaimsToTest) {
		    	claimNum++;
		    	String claimInfo = trainTextReader.nextLine();
				JSONObject claimJson = new JSONObject(claimInfo);
				String wikiName = "";
				String claim = (String) claimJson.get("claim");
				String filteredClaim = claim.replace(".", " .").replace(",", " ,").replace(".", " .").replace("\'s", " \'s").replace("(", "-lrb- ").replace(")", " -rrb-");
				String label = (String) claimJson.get("label");
				JSONArray answerEvidence = (JSONArray) claimJson.get("evidence");
				ArrayList<String> primaryEvidence = new ArrayList<String>();
				Map<String, Integer> wordMatch = new HashMap<String, Integer>();
				Map<String, Integer> wordMiss = new HashMap<String, Integer>();
				for(int i = 0; i < answerEvidence.length(); i++) {
					JSONArray evidenceSet = answerEvidence.getJSONArray(i);
					JSONArray primarySentence = evidenceSet.getJSONArray(0);
					boolean isAlone = (evidenceSet.length() == 1);
					if(primarySentence.length() == 5) {
						wikiName = primarySentence.get(2).toString();
						Object primarySentenceInfo = primarySentence.get(4);
						if(primarySentenceInfo != null) {
							//System.out.println(primarySentenceInfo.toString());
							int end = Math.max(primarySentenceInfo.toString().indexOf(" ."), Math.max(primarySentenceInfo.toString().indexOf(" ?"), primarySentenceInfo.toString().indexOf(" !")));
							if(end == -1 || !isAlone) { //TODO: handle cases with more than one sentence
								continue;
							}
							String primarySentenceText = primarySentenceInfo.toString().substring(0, end);
							if(!primaryEvidence.contains(primarySentenceText)) {
								primaryEvidence.add(primarySentenceText);
								String[] words = primarySentenceText.split(" ");
								for(int j = 0; j < words.length; j++) {
									String keyWord = words[j].toLowerCase();
									if(filteredClaim.toLowerCase().contains(keyWord)) {
										if(wordMatch.containsKey(keyWord)) {
											wordMatch.put(keyWord, wordMatch.get(keyWord) + 1);
										}
										else {
											wordMatch.put(keyWord, 1);
										}
									} 
								}
								String[] claimWords = filteredClaim.split(" ");
								for(int j = 0; j < claimWords.length - 1; j++) {
									String keyWord = claimWords[j].toLowerCase();
									if(!primarySentenceText.toLowerCase().contains(keyWord) && !wikiName.toLowerCase().contains(keyWord)) {
										if(wordMiss.containsKey(keyWord)) {
											wordMiss.put(keyWord, wordMiss.get(keyWord) + 1);
										}
										else {
											wordMiss.put(keyWord, 1);
										}
									} 
								}
							}
							
						}
						
					}
					
				}
				if(!label.equals("NOT ENOUGH INFO")) {
					writer.append("Claim: \"" + claim+"\"\n");
					writer.append("Label: \"" + label+"\"\n");
					writer.append("Wiki article: \"" + wikiName+"\"\n");
					writer.append("Primary supporting sentences: \n");
					for(String sentence: primaryEvidence) {
						writer.append(sentence+"\n");
					}
					writer.append("Matching words: \n");
					writer.append(wordMatch.toString() + "\n");
					writer.append("Missing words (not counting article topic): \n");
					writer.append(wordMiss.toString() + "\n");
					//System.out.println("named entities");
					writer.append("Named entities: \n");
					writer.append(getNamedEntities(claim, pipeline) + "\n");
					CoreDocument document = new CoreDocument(claim);
				    pipeline.annotate(document);
				    CoreSentence claimDoc = document.sentences().get(0);
					Tree constituencyTree = claimDoc.constituencyParse();
					//System.out.println("noun phrases");
					writer.append("Noun phrases: \n");
					writer.append(getMaxNounPhrases(claim, constituencyTree) + "\n");
					writer.append("\n");
				}
				System.out.print(".");
				
		    }
		    
		    
		    trainTextReader.close();
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
	
	private static ArrayList<String> getNamedEntities(String sentence, StanfordCoreNLP pipeline){
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
				namedEntities.add(namedEntity);
				namedEntity = "";
			}
		}
		if(neActive) {
			namedEntities.add(namedEntity);
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
	
	private static ArrayList<String> getMaxNounPhrases(String claim,  Tree constituencyTree) {
		
		ArrayList<String> nounPhrases = new ArrayList<String>();
		List<Tree> leaves = constituencyTree.getLeaves();
	    //Tree topicTree = leaves.get(topic.index()-1);
	    int phraseStart = 0;

	    //return ArrayList of topic phrases, from broadest to most narrow
	    ArrayList<Tree> nounTreeList = new ArrayList<Tree>();
	    String[] nounsAndPhrases = {"NN", "NNS", "NNP", "NNPS", "NP"};//, "VP", "CONJP"};
	    //boolean onlyNouns = true;
	    //boolean added = false;
	    //System.out.println("noun phrases 1");
	    while(phraseStart < leaves.size()) {
	    	Tree nounTree = leaves.get(phraseStart);
	    	Tree nounPhrase = nounTree.deepCopy();
	    	//System.out.println("noun phrases 2");
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
		    nounPhrases.add(nounPhrase);
	    }

	    
	    return nounPhrases;
	}

}

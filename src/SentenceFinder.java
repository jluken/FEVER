import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

import java.nio.file.Path;
import java.nio.file.Paths;
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
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
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
	static String outputFileName = "found_sentences.jsonl";
	static String wikiDirName = "wiki-dump";

	static int numClaimsToTest = 10;
	static int claimBatchSize = 100;
	static boolean testAll = false;

	
	static Map<String, Map<String, Object>> wikiMap;
	//static Map<String, ArrayList<String>> disambiguationMap;
	//static Map<String, String> lowercaseMap;

    static IDictionary synonymDict;

	public static void main(String[] args) {
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss");  
	    System.out.println("Beginning document processing. Time: "+dtf.format(LocalDateTime.now()));	    
		
		StanfordCoreNLP pipeline = establishPipeline();
	    System.out.println("CoreNLP pipeline established. Time: "+dtf.format(LocalDateTime.now()));	    
	    getWikiMaps(wikiDirName);
		System.out.println("wikiMaps compiled. Time: "+dtf.format(LocalDateTime.now()));
		getSynDict();
		

		int claimCount =0;
		try {
			Scanner claimReader = new Scanner(new FileReader(claimsFileName));
			File oldFile = new File(outputFileName);
			if(oldFile.exists()) {
				oldFile.delete();
			}
			BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName, true));
			
			
			while(claimReader.hasNext() && (testAll || claimCount < numClaimsToTest)) {
				claimCount++;
				try {
					String claimInfo = claimReader.nextLine();
					JSONObject claimJson = new JSONObject(claimInfo);
					String claim = Normalizer.normalize(claimJson.getString("claim"), Normalizer.Form.NFC);   
					int id = claimJson.getInt("id");
					System.out.println("Claim " + claimCount + ": " +claim + " Time: " + dtf.format(LocalDateTime.now()));
					
					CoreDocument document = new CoreDocument(claim);
					pipeline.annotate(document);
					CoreSentence claimDoc = document.sentences().get(0);
					Tree constituencyTree = claimDoc.constituencyParse();
					SemanticGraph dependencyGraph = claimDoc.dependencyParse();
					String formattedClaim = formatSentence(claim);
					ArrayList<String[]> claimNE = getNamedEntities(formattedClaim, pipeline);
					ArrayList<Map<String, String>> documents = findGivenDoc(claimInfo);
					//Map<String, Object> documents = findDocuments(claim, dependencyGraph, constituencyTree, claimNE);
					
					Map<String, ArrayList<Object[]>> evidenceSentences = findSentences(pipeline, claim, dependencyGraph, constituencyTree, claimNE, documents);
				    
				    
				    String JSONStr = evidenceToLine(id, claim, evidenceSentences);
				    writer.append(JSONStr);
				    writer.append("\n");
				}catch(Exception e){
					e.printStackTrace();
					System.out.println("Something went wrong with processeing claim " + claimCount + ". Skipping");
				    writer.append("\n");
				}
				if(claimCount % claimBatchSize == 0) {
					writer.close();
					writer = new BufferedWriter(new FileWriter(outputFileName, true));
				}
			    
			}
			claimReader.close();
			writer.close();
		}catch(Exception e) {
			e.printStackTrace();
		}

		
	}
	
	private static StanfordCoreNLP establishPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, parse, depparse, lemma, ner");
	    props.setProperty("coref.algorithm", "neural");
	    props.put("ner.model", "english.conll.4class.distsim.crf.ser.gz");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}
	
	private static void getSynDict() {
		try {
			synonymDict = new Dictionary (new URL ("file", null , "dict"));
			synonymDict.open();
		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
        
	}
	
	private static ArrayList<Map<String, String>> findGivenDoc(String answerInfo){
		ArrayList<Map<String, String>> docMap = new ArrayList<Map<String, String>>();
		JSONObject answerJson;
		try {
			answerJson = new JSONObject(answerInfo);
			JSONArray answerEvidence = (JSONArray) answerJson.get("evidence");
			String wikiName = answerEvidence.getJSONArray(0).getJSONArray(0).get(2).toString();
			wikiName = Normalizer.normalize(wikiName, Normalizer.Form.NFC);
			if(wikiName.equals("null")) {
				return docMap;
			}
			ArrayList<String> wikiArrayList= new ArrayList<String>();
			wikiArrayList.add(wikiName);
			docMap = getBackupDocs(wikiArrayList);
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return docMap;
	}
	
	private static Map<String, ArrayList<Object[]>> findSentences(StanfordCoreNLP pipeline, String claim, SemanticGraph dependencyGraph, 
			Tree constituencyTree, ArrayList<String[]> namedEntities, ArrayList<Map<String, String>> wikis){
		Map<String, ArrayList<Object[]>> evidenceSentences = new HashMap<String, ArrayList<Object[]>>();
		String root = dependencyGraph.getFirstRoot().originalText().toLowerCase();
		for(Map<String, String> wiki : wikis) {
			ArrayList<Object[]> wikiSents = new ArrayList<Object[]>();
			String[] wikiLines = wiki.get("lines").split("\\n\\d*\\t");
			String wikiName = Normalizer.normalize(wiki.get("id"), Normalizer.Form.NFC);
			String wikiTitle = formatSentence(wikiName.replace("_", " ")).toLowerCase();
			List<String[]> nane = getNounsAndNamedEntities(wikiTitle, constituencyTree, namedEntities);
			for(int i = 0; i < wikiLines.length; i++) {
				String sentence = getSentenceTextFromWikiLines(wikiLines[i]);
				if(containsNamedEntities(sentence, claim, nane, wikiTitle, pipeline, root) || 
						containsValidRoot(sentence, root, pipeline)) {
					Object[] evidence = new Object[2];
					evidence[0] = i;
					evidence[1] = sentence;
					wikiSents.add(evidence);
				}	
			}
			if(!wikiSents.isEmpty()) {
				evidenceSentences.put(wikiName, wikiSents);
			}
		}
		return evidenceSentences;
	}
	
	private static String evidenceToLine(int id, String claim, Map<String, ArrayList<Object[]>> evidenceSentences) {
		ArrayList<JSONArray> evidenceSentencesJSON = new ArrayList<JSONArray>();
		for(String wiki: evidenceSentences.keySet()) {
			ArrayList<Object[]> evidenceSets = evidenceSentences.get(wiki);
			for(Object[] evidenceSet : evidenceSets) {
				Object[] evidenceArr = {wiki, evidenceSet[0], evidenceSet[1]};
				ArrayList<JSONArray> evidenceSetJSON = new ArrayList<JSONArray>();
				evidenceSetJSON.add(new JSONArray(Arrays.asList(evidenceArr)));
				evidenceSentencesJSON.add(new JSONArray(evidenceSetJSON));
			}
		}
		JSONArray evidence = new JSONArray(evidenceSentencesJSON);
		Map<String, Object> evidenceMap = new HashMap<String, Object>();
		evidenceMap.put("id", id);
		evidenceMap.put("claim", claim);
		evidenceMap.put("evidence", evidence);
		return new JSONObject(evidenceMap).toString();
	}
	
	private static List<String> lemmatize(StanfordCoreNLP pipeline, String text) {
        List<String> lemmas = new ArrayList<String>();
        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        for(CoreMap sentence: sentences) {
            for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                lemmas.add(token.get(LemmaAnnotation.class));
            }
        }

        return lemmas;
    }
	
	private static List<String> getSynonyms (StanfordCoreNLP pipeline, String word, POS pos, IDictionary dict){
		List<String> syns = new ArrayList<String>();
		String lemma = word;
		try {
			 lemma = lemmatize(pipeline, word).get(0);
			 IIndexWord idxWord = dict.getIndexWord (lemma, pos) ;
			 IWordID wordID = idxWord.getWordIDs().get(0) ;
			 IWord iword = dict.getWord(wordID);
			 ISynset synset = iword.getSynset();
			 for (IWord syn : synset.getWords()) {
				 syns.add(syn.getLemma().replace("_", " "));
		        }
		}catch(Exception e){
			syns.add(lemma);
		}
		 return syns;
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
	
	private static String formatSentence(String sentence) {
		String newSent = sentence.replace(",", " ,").replace(".", " .").replace(";", " ;").replace(":", " :").replace("'s", " 's").replace("' ", " ' ");
		newSent = newSent.replace("-LRB-", "-LRB- ").replace("-RRB-", " -RRB-").replace("-RSB-", " -RSB-").replace("-LSB-", " -LSB-");
		newSent = newSent.replace("(", "-LRB- ").replace(")", " -RRB-").replace("]", " -RSB-").replace("[", " -LSB-");
		return newSent;
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
		String[] ignoredNouns = {"something", "anything", "thing", "there", "name", "part"};
		List<String> ignoredNounsList = Arrays.asList(ignoredNouns);
		List<Tree> leaves = constituencyTree.getLeaves();
	    int wordNum = 0;

	    ArrayList<String> wordList = new ArrayList<String>();
	    String[] nounsTags = {"NN", "NNS", "NNP", "NNPS", "NP"};
	    while(wordNum < leaves.size()) {
	    	Tree word = leaves.get(wordNum);
	    	Tree type = word.parent(constituencyTree);
	    	String wordStr = word.toString();
	    	if(Arrays.asList(nounsTags).contains(type.label().toString()) && !wikiTitle.contains(wordStr.toLowerCase()) 
	    			&& !ignoredNounsList.contains(wordStr)) {
	    		wordList.add(wordStr.toLowerCase());
	    	} 
	    	wordNum++;
	    }
    
	    return wordList;
	}

	private static ArrayList<String[]> getNounsAndNamedEntities(String wikiTitle,  Tree constituencyTree, List<String[]> namedEntities){
		ArrayList<String> nouns = getNouns(wikiTitle, constituencyTree);
	    List<String[]> filteredNamedEntities = namedEntities.stream().filter(arr -> !wikiTitle.contains(arr[0]))
				.collect(Collectors.toList());
			
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

	private static boolean containsNamedEntities(String evidenceSentence, String claim, List<String[]> claimNE, String wikiTitle, StanfordCoreNLP pipeline, String root) {
		Map<String, String> altMap = new HashMap<String, String>();
		altMap.put("NATIONALITY", "COUNTRY");
		altMap.put("COUNTRY", "NATIONALITY");
		altMap.put("DATE", "SET");
		altMap.put("SET", "DATE");
		
		boolean validSentence = true;
		if(evidenceSentence.isEmpty() || claimNE.isEmpty()) {
			return false;
		}

		String[] entityToBeSwapped = {null, null};
		for(String[] ne: claimNE) {
			if(!evidenceSentence.toLowerCase().contains(ne[0]) && entityToBeSwapped[1] == null) {
				entityToBeSwapped = ne.clone();
			}
			else if(!evidenceSentence.toLowerCase().contains(ne[0])) {
				validSentence = false;
			}	
		}
		if(validSentence && entityToBeSwapped[1] == "NOUN" && claimNE.size() < 3) {
			//both sentences need to be noun complements, have at least 2 other matching entities
			//or the evidence needs to have a synonym of the missing noun
			if(!isNounComplement(claim) || !isNounComplement(evidenceSentence)) {
				validSentence = containsSynonym(entityToBeSwapped[0], POS.NOUN, evidenceSentence, pipeline);
				if(!validSentence) {
				}
			}
		}	
		
		List<String[]> evidenceEntities = null;
		if(validSentence && entityToBeSwapped[1] != "NOUN" && entityToBeSwapped[1] != null) {
			if(isVerb(root, pipeline) && !(evidenceSentence.contains(root) || containsSynonym(root, POS.VERB, evidenceSentence, pipeline))) {
				//if the evidence is missing a root verb and a named entity, then the evidence needs to have a synonym of the missing root
				validSentence = false;
			}
			else {
				//otherwise the entity can be swapped out for one of the same type
				evidenceEntities = getNamedEntities(evidenceSentence, pipeline).stream().filter(arr -> !wikiTitle.contains(arr[0]))
						.collect(Collectors.toList());
				validSentence = false;
				String alternative = altMap.get(entityToBeSwapped[1]);
				for(String[] ne: evidenceEntities) {
					if(ne[1].equals(entityToBeSwapped[1]) || (alternative != null && alternative.equals(ne[1]))) {
						validSentence = true;
					}
				}
			}
		}

		if(!validSentence &&  evidenceSentence.contains("-LRB-") && 
				(claim.contains("born") || claim.contains("died") || claim.contains("dead"))) {
			//if the sentence relates to birth or death, we can check to see if the sentence has wikipedia-formatted birth/death info 
			if(evidenceEntities == null) {
				evidenceEntities = getNamedEntities(evidenceSentence, pipeline).stream().filter(arr -> !wikiTitle.contains(arr[0]))
					.collect(Collectors.toList());
			}
			validSentence = wikiBirthDeath(claim, evidenceEntities, evidenceSentence);
		}

		return validSentence;
	}
	
	private static boolean containsSynonym(String word, POS pos, String sentence, StanfordCoreNLP pipeline) {
		boolean contains = false;
		String[] ignoredLemmas = {"have", "do", "be"};
		List<String> sentLemmas = lemmatize(pipeline, sentence);
		List<String> syns = getSynonyms(pipeline, word, pos, synonymDict);
		for(String syn : syns) {
			if(sentLemmas.contains(syn) && !Arrays.asList(ignoredLemmas).contains(syn)) {
				contains = true;
			}
		}
		return contains;
	}
	
	private static boolean isNounComplement(String sentence) {
		String[] bes = {" is", " was"};
		String[] adverbs = {" ", " only ", " always ", " never ", " not ", };
		String[] dets = {"a ", "an ", "the "};
		for(String verb : bes) {
			for(String adverb : adverbs) {
				for(String det : dets) {
					String ncTag = verb + adverb + det;
					if(sentence.contains(ncTag)) {
						return true;
					}
				}
			}
		}
		return false;
	}
	
	private static boolean containsValidRoot(String sentence, String root, StanfordCoreNLP pipeline) {
		String[] isWords = {"is", "was", "be", "are", "were", "has", "had", "have"};
		String rootAlone = " " + root + " ";
		boolean validRoot = false;
		if(isVerb(root, pipeline) && !Arrays.asList(isWords).contains(root) && 
				(sentence.toLowerCase().contains(rootAlone) || sentence.toLowerCase().startsWith(root))) {
			validRoot = true;
		}
		return validRoot;
	}
	
	private static boolean wikiBirthDeath(String claim, List<String[]> SentNameEntities, String sentence) {
		boolean edgeCase = false;
		List<String> dates = SentNameEntities.stream().filter(ne -> ne[1].equals("DATE")).map(ne -> ne[0]).collect(Collectors.toList());
		for(String date : dates) {
			boolean parenDate = Pattern.compile("lrb(.*)" + date + "(.*)rrb").matcher(sentence.toLowerCase()).find();
			if(parenDate) {
				edgeCase = true;
			}
		}
		
		return edgeCase;
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
	
	
	private static ArrayList<Map<String, String>> getBackupDocs(ArrayList<String> backupDocumentKeys){
		ArrayList<Map<String, String>> backupDocs = new ArrayList<Map<String, String>>();
		for(String key : backupDocumentKeys) {
			try {
				Map<String, String> backupDoc = new HashMap<String, String>();
				Map<String, Object> fileInfo = wikiMap.get(key);
                Path wikiDirPath = Paths.get(wikiDirName);
                Path wikiListPath =  wikiDirPath.resolve(Paths.get((String)fileInfo.get("fileName")));
                String wikiListName = wikiListPath.toString();

				BufferedReader reader = new BufferedReader(new FileReader(wikiListName));
				Long byteOffset = (Long) fileInfo.get("offset");
				reader.skip(byteOffset);
			    String wikiEntry = reader.readLine();
			    JSONObject wikiJson = new JSONObject(wikiEntry);
			    backupDoc.put("id", wikiJson.getString("id"));
			    backupDoc.put("text", wikiJson.getString("text"));
			    backupDoc.put("lines", wikiJson.getString("lines"));
			    backupDocs.add(backupDoc);
			    reader.close();
			} catch (FileNotFoundException e) {
				System.out.println("Could not open file  "+ key);
				e.printStackTrace();
			} catch (JSONException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return backupDocs;
	}
	
	private static void getWikiMaps(String wikiDirName) {
		wikiMap = new HashMap<String, Map <String, Object>>();
//		disambiguationMap = new HashMap<String, ArrayList<String>>();
//		lowercaseMap = new HashMap<String, String>();
		
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
						String wikiEntry = Normalizer.normalize(s.nextLine(), Normalizer.Form.NFC);
					    JSONObject wikiJson = new JSONObject(wikiEntry);
					    String id = wikiJson.getString("id");
					    if(!id.isEmpty()) {
					    	String fileName = wikiEntryList.getName();
					    	Map<String, Object> docLocation = new HashMap<String, Object>();
					    	docLocation.put("fileName", fileName);
					    	docLocation.put("offset", byteOffset);
							wikiMap.put(id, docLocation);
//							lowercaseMap.put(id.toLowerCase(), id);
//							
//							int paren = id.indexOf("-LRB-");
//							if(paren > 0 && !id.contains("disambiguation")) {
//								String base = id.substring(0, paren-1);
//								ArrayList<String> disambiguationChildren = new ArrayList<String>();
//								if(disambiguationMap.containsKey(base)) {
//									disambiguationChildren = disambiguationMap.get(base);
//									disambiguationChildren.add(id);
//									disambiguationMap.put(base, disambiguationChildren);
//								}
//								else {
//									disambiguationChildren.add(id);
//									disambiguationMap.put(base, disambiguationChildren);
//								}
//							}
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
	
}
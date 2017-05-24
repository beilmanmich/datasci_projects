#TOP 100 WORDS MR_JOB - WikiData

from mrjob.job import MRJob
from mrjob.step import MRStep
import xml.etree.ElementTree as ET
import re, heapq, mwparserfromhell

WORD_RE = re.compile(r"\w+")
START_RE = re.compile('.*<page>.*')
END_RE= re.compile('.*</page>.*')

class MRTOP100(MRJob):
    
    def string_mapper_init(self):
        self.pageall=''
        
    def string_mapper(self,_,line):
        self.pageall=self.pageall+line
        if END_RE.match(line):
            page=self.pageall
            self.pageall=''
            if START_RE.match(page):
                yield (None,page)

    def string_reducer(self,_,pages):
        for page in pages:
            yield (None,page)
    
    def mapper_get_xml_words(self, _, page):
        root = ET.fromstring(page)
        tag_and_text = [(x.tag, x.text) for x in root.getiterator()]
        for tag, text in tag_and_text:
            if (tag == 'text' and text):
                hell_filter = mwparserfromhell.parse(text)
                for parsedtext in hell_filter.filter_text():    
                    for word in WORD_RE.findall(parsedtext.value):
                        yield (word.lower(), 1)

    def combiner_count_words(self, word, counts):
        # sum the words we've seen so far                                                            
        yield (word, sum(counts))

    def reducer_count_words(self, word, counts):
        # send all (num_occurrences, word) pairs to the same reducer.                                
        # num_occurrences is so we can easily use Python's max() function.                           
        yield (word, sum(counts))

    def heap_mapper_init(self):
        self.h = []

    def heap_mapper(self,word,counts):
        #pushes the value of (counts,word) onto self.h                                               
        heapq.heappush(self.h,(counts,word))

    def heap_mapper_final(self):
        #returns a list of the 100 largest elements from the self.h dataset                          
        largest = heapq.nlargest(100,self.h)
        for count, word in largest:
            yield (None, (count,word))

    def heap_reducer_init(self):
        self.h_combined = []

    def heap_reducer(self, _, word_counts):
        #pushes the value of counts (words[0]), word (words[1]) onto self.h_combined                       
        for words in word_counts:
            heapq.heappush(self.h_combined, (words[0],words[1]))

    def heap_reducer_final(self):
        largest = heapq.nlargest(100,self.h_combined)
        words = [(word,  int(count)) for count,word in largest]
        yield (None, words)

    def steps(self):
        return [
            MRStep(mapper_init=self.string_mapper_init,
                   mapper=self.string_mapper,
                   reducer=self.string_reducer),
            MRStep(mapper=self.mapper_get_xml_words,
                   combiner=self.combiner_count_words,
                   reducer=self.reducer_count_words),
            MRStep(mapper_init = self.heap_mapper_init,
                   mapper = self.heap_mapper,
                   mapper_final = self.heap_mapper_final,
                   reducer_init = self.heap_reducer_init,
                   reducer = self.heap_reducer,
                   reducer_final = self.heap_reducer_final)]
        
if __name__ == '__main__':
    MRTOP100.run()
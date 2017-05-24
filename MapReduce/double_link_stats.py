#MRJob Double_Link_Stats WikiData

from mrjob.job import MRJob
from mrjob.step import MRStep
from math import sqrt
import xml.etree.ElementTree as ET
import re, heapq, mwparserfromhell, random

WORD_RE = re.compile(r"\w+")
START_RE = re.compile('.*<page>.*')
END_RE = re.compile('.*</page>.*')
PAGES_NOT_WANTED = ['list\sof.*','category:','talk:','file:','files:','help:','template:']

def is_valid_link(link):
    return not any([x in link.lower() for x in PAGES_NOT_WANTED])


class MRDOUBLELINKS(MRJob):
    
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
                               
    def mapper_xml(self,_,page):
        try:
            root = ET.fromstring(page.encode('utf-8'))
        except:
            root = None
        if root is not None:
            try:
                link_title = root.find('title').text #this page's link title
            except:
                pass
            if is_valid_link(link_title):
                for element in root.findall('.//text'):
                    try:
                        if element.text is not None:
                            hell_filter = mwparserfromhell.parse(element.text)
                            wikilinks = hell_filter.filter_wikilinks()
                            links = [str(x.title) for x in wikilinks if is_valid_link(x.title) and x.title != link_title]
                            links = list(set(links))
                            for link in links:
                                i = link_title
                                j = link
                                weight = 1./(len(links) + 10.)
                                yield i, ('bycol',j,weight)
                                yield j, ('byrow',i,weight)
                    except:
                        pass
                                        
    def expand_reducer(self, _, values):
        values = [x for x in values] # converts values from iterator to list
        byrow = [(i, weight) for (by, i, weight) in values if by == "byrow"]
        bycol = [(k, weight2) for (by, k, weight2) in values if by == "bycol"]
        for i, weight in byrow:
            for k, weight2 in bycol:
                if i != k: #don't want self-loops
                    yield ((min(i,k),max(i,k)), weight * weight2)
                    
    def sum_reducer(self, k, values):
        yield (k, sum(values)/2.)
           
    def heap_mapper_init(self):
        self.h = []

    def heap_mapper(self,word,counts):
        # pushes the value of (counts, words) onto self.h                          
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
            MRStep(mapper_init = self.string_mapper_init,
                   mapper = self.string_mapper,
                   reducer = self.string_reducer),
            MRStep(mapper = self.mapper_xml,
                   reducer = self.expand_reducer),
            MRStep(reducer=self.sum_reducer),
            MRStep(mapper_init = self.heap_mapper_init,
                   mapper = self.heap_mapper,
                   mapper_final = self.heap_mapper_final,
                   reducer_init = self.heap_reducer_init,
                   reducer = self.heap_reducer,
                   reducer_final = self.heap_reducer_final)]

if __name__ == '__main__':
    MRDOUBLELINKS.run()
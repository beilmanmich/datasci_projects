#MRJOB LinkStats WikiData

from mrjob.job import MRJob
from mrjob.step import MRStep
from math import sqrt
import xml.etree.ElementTree as ET
import re, heapq, mwparserfromhell, random


WORD_RE = re.compile(r"\w+")
START_RE = re.compile('.*<page>.*')
END_RE = re.compile('.*</page>.*')
RESERVOIR_LEN = 2000


class MRLINKSTATS(MRJob):
    
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
            
    def mapper_xml_init(self):
        self.sum_x = 0
        self.sum_x2=0
        self.pagecount=0
        self.reservoir = []
        self.k = RESERVOIR_LEN
                   
    def mapper_xml(self,_,page):
        try:
            root = ET.fromstring(page)
        except:
            root = None
        if root is not None:
            tag_and_text = [(x.tag, x.text) for x in root.getiterator()]
            for tag, text in tag_and_text:
                try:
                    if (tag == 'text' and text):
                        hell_filter = mwparserfromhell.parse(text)
                        wikilinks = hell_filter.filter_wikilinks()
                        x = len(set(wikilinks))
                        randno = random.random()
                        self.pagecount += 1
                        self.sum_x += x
                        self.sum_x2 += x**2
                        if len(self.reservoir) < self.k:
                            heapq.heappush(self.reservoir,(randno,x))
                        elif randno > self.reservoir[0][0]:
                            heapq.heapreplace(self.reservoir,(randno,x))
                except:
                    pass
                        
    def mapper_xml_final(self):
        yield('mean_and_deviation',(self.sum_x,self.sum_x2,self.pagecount))
        for randno,x in self.reservoir:
            yield ('reservoir',(randno,x))
      
    def reducer_compute_stats(self, key, vals):
        if key == 'mean_and_deviation':
            N=0
            sum_x = 0
            sum_x_sq = 0
            for val in vals:
                x,x_sq,x_num = val
                N += x_num
                sum_x += x
                sum_x_sq += x_sq
            N = float(N)
            mean = sum_x/N
            std_dev = sqrt(sum_x_sq/N - mean**2)
            yield ('page count',int(N))
            yield ('mean',mean)
            yield ('std dev',std_dev)
        elif key == 'reservoir':
            counts = [x for randno,x in vals]
            counts.sort()
            yield ('5th percentile', float(counts[int(round(len(counts)*0.05)-1)]))
            yield ('25th percentile', float(counts[int(round(len(counts)*0.25)-1)]))
            yield ('median', float(counts[int(round(len(counts)*0.5)-1)]))
            yield ('75th percentile', float(counts[int(round(len(counts)*0.75)-1)]))
            yield ('95th percentile', float(counts[int(round(len(counts)*0.95)-1)]))
                    
    def steps(self):
        return [
            MRStep(mapper_init = self.string_mapper_init,
                    mapper = self.string_mapper,
                    reducer = self.string_reducer),
            MRStep(mapper_init = self.mapper_xml_init,
                   mapper = self.mapper_xml,
                   mapper_final = self.mapper_xml_final,
                   reducer=self.reducer_compute_stats)]
  
                   
if __name__ == '__main__':
    MRLINKSTATS.run()
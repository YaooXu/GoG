from dotenv import load_dotenv
load_dotenv()

import json
import os
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQLPATH = os.environ['SPARQLPATH']

def test():
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql_txt = """PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?name3
            WHERE {
            ns:m.0k2kfpc ns:award.award_nominated_work.award_nominations ?e1.
            ?e1 ns:award.award_nomination.award_nominee ns:m.02pbp9.
            ns:m.02pbp9 ns:people.person.spouse_s ?e2.
            ?e2 ns:people.marriage.spouse ?e3.
            ?e2 ns:people.marriage.from ?e4.
            ?e3 ns:type.object.name ?name3
            MINUS{?e2 ns:type.object.name ?name2}
            }
        """
        #print(sparql_txt)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        print(results)
    except:
        print('Your database is not installed properly !!!')

test()

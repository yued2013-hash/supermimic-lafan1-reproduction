import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_results import validate_evaluation,verify

class Evidence(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((ROOT/'results/lafan1_seed0/eval_fp32.json').read_text())
    def test_recorded_experiment_is_complete(self):
        self.assertEqual(verify()['clips'],40)
    def test_duplicate_sequence_rejected(self):
        self.data['clips'][1]['name']=self.data['clips'][0]['name']
        with self.assertRaises(ValueError):validate_evaluation(self.data)
    def test_shortened_sequence_rejected(self):
        self.data['clips'][0]['steps']-=1
        with self.assertRaises(ValueError):validate_evaluation(self.data)
    def test_inconsistent_summary_rejected(self):
        self.data['summary']['mean_ret_paper']+=.01
        with self.assertRaises(ValueError):validate_evaluation(self.data)
    def test_nonfinite_return_rejected(self):
        self.data['clips'][0]['ret_paper']=float('nan')
        with self.assertRaises(ValueError):validate_evaluation(self.data)

if __name__=='__main__':unittest.main()

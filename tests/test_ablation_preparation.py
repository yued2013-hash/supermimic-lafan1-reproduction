import ast,importlib.util,sys,unittest,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_horizon_source import transform,ANCHOR

class SourcePatch(unittest.TestCase):
    def test_divisor_fixed_not_horizon_dependent(self):
        tree=ast.parse(transform('def update(J,H):\n    '+ANCHOR+'\n    return loss_pi\n'))
        division=tree.body[0].body[0].value
        self.assertEqual(division.right.value,32.0)
    def test_unexpected_or_repeated_source_rejected(self):
        for text in ['def wrong(): pass',ANCHOR+'\n'+ANCHOR]:
            with self.assertRaises(ValueError):transform(text)
    def test_already_patched_source_rejected(self):
        with self.assertRaises(ValueError):transform(transform('def update(J,H):\n    '+ANCHOR+'\n'))
    def test_diagnostic_is_not_a_training_motion_library(self):
        from run_horizon import validate_motion_container
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wrong.npz'
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('states.npy',b'placeholder')
            with self.assertRaises(ValueError):validate_motion_container(path)

@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional diagnostic-data tests require numpy')
class DiagnosticCoordinates(unittest.TestCase):
    def test_local_angular_velocity_converted_to_world(self):
        import numpy as np
        from prepare_wm_diagnostic import to_state
        qp=np.zeros((1,36));qv=np.zeros((1,35))
        qp[0,3:7]=[2**-.5,0,0,2**-.5];qv[0,3]=1
        state=to_state(qp,qv)
        np.testing.assert_allclose(state[0,39:42],[0,1,0],atol=1e-6)
    def test_invalid_quaternion_rejected(self):
        import numpy as np
        from prepare_wm_diagnostic import to_state
        with self.assertRaises(ValueError):to_state(np.zeros((1,36)),np.zeros((1,35)))

if __name__=='__main__':unittest.main()

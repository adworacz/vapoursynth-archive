import unittest
import vapoursynth as vs

class CoreTestSequence(unittest.TestCase):
    
    def setUp(self):
        self.core = vs.get_core(threads=10)
   
    def test_num_threads(self):
        self.assertEqual(self.core.num_threads, 10)
        
    def test_func1(self):
        with self.assertRaises(vs.Error):
            self.core.blah.list_functions()
            
    def test_arg1(self):
        self.core.std.BlankClip()

    def test_arg2(self):
        self.core.std.BlankClip(width=50)
        
    def test_arg3(self):
        self.core.std.BlankClip(width=[50])
        
    def test_arg4(self):
        with self.assertRaises(vs.Error):
            self.core.std.BlankClip(width=[50, 50])
        
    def test_arg5(self):
        with self.assertRaises(vs.Error):
            self.core.std.BlankClip(width=[])
        
    def test_arg6(self):
        self.core.std.BlankClip(_width=50)
        
    def test_arg7(self):
        with self.assertRaises(vs.Error):
            self.core.std.BlankClip(_width=[])
        
    def test_arg8(self):
        with self.assertRaises(vs.Error):
            self.core.std.BlankClip(10,10,10,10,10,10,10,10,10,10,10,10,10,10)
  
    def test_arg9(self):
        with self.assertRaises(vs.Error):
            self.core.std.BlankClip(width2=50)

    def test_arg10(self):
        with self.assertRaises(vs.Error):
            self.core.std.FlipVertical()
        
if __name__ == '__main__':
    unittest.main()

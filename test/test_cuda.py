import unittest
import vapoursynth as vs


class CoreTestSequence(unittest.TestCase):

    def setUp(self):
        self.core = vs.Core()

    def testMergeDifference(self):
        clip1 = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115])
        clip2 = self.core.std.BlankClip(format=vs.YUV420P8, color=[113, 115, 115])

        cpu = self.core.std.Merge(clips=[clip1, clip2])

        clip1 = self.core.std.TransferFrame(clip1, 1)
        clip2 = self.core.std.TransferFrame(clip2, 1)
        gpu = self.core.std.Merge(clips=[clip1, clip2])
        gpu = self.core.std.TransferFrame(gpu, 0)

        diff = self.core.std.PlaneDifference([cpu, gpu], 0, prop="PlaneDifference0")
        diff = self.core.std.PlaneDifference([diff, gpu], 1, prop="PlaneDifference1")
        diff = self.core.std.PlaneDifference([diff, gpu], 2, prop="PlaneDifference2")

        for i in range(diff.num_frames):
            frame = diff.get_frame(i)
            self.assertEqual(frame.props.PlaneDifference0[0], 0)
            self.assertEqual(frame.props.PlaneDifference1[0], 0)
            self.assertEqual(frame.props.PlaneDifference2[0], 0)


if __name__ == '__main__':
    unittest.main()

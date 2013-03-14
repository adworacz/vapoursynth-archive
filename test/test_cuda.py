import unittest
import vapoursynth as vs


class CoreTestSequence(unittest.TestCase):

    def setUp(self):
        self.core = vs.Core()

    def checkDifference(self, cpu, gpu):
        diff = self.core.std.PlaneDifference([cpu, gpu], 0, prop="PlaneDifference0")
        diff = self.core.std.PlaneDifference([diff, gpu], 1, prop="PlaneDifference1")
        diff = self.core.std.PlaneDifference([diff, gpu], 2, prop="PlaneDifference2")

        for i in range(diff.num_frames):
            frame = diff.get_frame(i)
            self.assertEqual(frame.props.PlaneDifference0[0], 0)
            self.assertEqual(frame.props.PlaneDifference1[0], 0)
            self.assertEqual(frame.props.PlaneDifference2[0], 0)

    def testLutDifference(self):
        clip = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115])

        luty = []
        for x in range(2 ** clip.format.bits_per_sample):
            luty.append(max(min(x, 235), 16))
        lutuv = []
        for x in range(2 ** clip.format.bits_per_sample):
            lutuv.append(max(min(x, 100), 16))

        cpu = self.core.std.Lut(clip=clip, lut=luty, planes=0)
        cpu = self.core.std.Lut(clip=cpu, lut=lutuv, planes=[1, 2])

        clip = self.core.std.TransferFrame(clip, 1)
        gpu = self.core.std.Lut(clip=clip, lut=luty, planes=0)
        gpu = self.core.std.Lut(clip=gpu, lut=lutuv, planes=[1, 2])
        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)


    def testMergeDifference(self):
        clip1 = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115])
        clip2 = self.core.std.BlankClip(format=vs.YUV420P8, color=[113, 115, 115])

        cpu = self.core.std.Merge(clips=[clip1, clip2])

        clip1 = self.core.std.TransferFrame(clip1, 1)
        clip2 = self.core.std.TransferFrame(clip2, 1)
        gpu = self.core.std.Merge(clips=[clip1, clip2])
        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)


if __name__ == '__main__':
    unittest.main()

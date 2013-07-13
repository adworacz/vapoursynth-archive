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

    def testAddBorders(self):
        gpu = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115], gpu=1)
        cpu = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115])

        cpu = self.core.std.AddBorders(cpu, left=16, right=32, top=64, bottom=128, color=[115, 242, 69])
        gpu = self.core.std.AddBorders(gpu, left=16, right=32, top=64, bottom=128, color=[115, 242, 69])

        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)

    def testBlankClip(self):
        cpu = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115])
        gpu = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115], gpu=1)

        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)

    def testBlankClip16bit(self):
        cpu = self.core.std.BlankClip(format=vs.YUV420P16, color=[300, 700, 900])
        gpu = self.core.std.BlankClip(format=vs.YUV420P16, color=[300, 700, 900], gpu=1)

        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)

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

    def testLutDifference16bit(self):
        clip = self.core.std.BlankClip(format=vs.YUV420P10, color=[300, 700, 900])

        luty = []
        for x in range(2 ** clip.format.bits_per_sample):
            luty.append(max(min(x, 800), 300))
        lutuv = []
        for x in range(2 ** clip.format.bits_per_sample):
            lutuv.append(max(min(x, 800), 300))

        cpu = self.core.std.Lut(clip=clip, lut=luty, planes=0)
        cpu = self.core.std.Lut(clip=cpu, lut=lutuv, planes=[1, 2])

        clip = self.core.std.TransferFrame(clip, 1)
        gpu = self.core.std.Lut(clip=clip, lut=luty, planes=0)
        gpu = self.core.std.Lut(clip=gpu, lut=lutuv, planes=[1, 2])
        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)

    def testTransposeDifference(self):
        cpu = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115])
        gpu = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115], gpu=1)

        cpu = self.core.std.Transpose(cpu)
        gpu = self.core.std.Transpose(gpu)

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

    def testMaskedMergeDifference(self):
        clip1 = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 242, 115])
        clip2 = self.core.std.BlankClip(clip=clip1, color=[113, 115, 115])
        mask = self.core.std.BlankClip(clip=clip1, color=[235, 235, 235])

        cpu = self.core.std.MaskedMerge(clips=[clip1, clip2], mask=mask)

        clip1 = self.core.std.TransferFrame(clip1, 1)
        clip2 = self.core.std.TransferFrame(clip2, 1)
        mask = self.core.std.TransferFrame(mask, 1)
        gpu = self.core.std.MaskedMerge(clips=[clip1, clip2], mask=mask)
        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)

    def testExprDifference(self):
        clipa = self.core.std.BlankClip(format=vs.YUV420P8, color=[112, 112, 220])
        clipb = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 45, 73])
        clipc = self.core.std.BlankClip(format=vs.YUV420P8, color=[200, 119, 182])

        gpua = self.core.std.TransferFrame(clipa, 1)
        gpub = self.core.std.TransferFrame(clipb, 1)
        gpuc = self.core.std.TransferFrame(clipc, 1)

        cpu = self.core.std.Expr(clips=[clipa, clipb, clipc], expr=["x y + z + 3 /"])
        gpu = self.core.std.Expr(clips=[gpua, gpub, gpuc], expr=["x y + z + 3 /"])

        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)

    def testExprDifferenceLargeOps(self):
        clipa = self.core.std.BlankClip(format=vs.YUV420P8, color=[112, 112, 220])
        clipb = self.core.std.BlankClip(format=vs.YUV420P8, color=[69, 45, 73])

        gpua = self.core.std.TransferFrame(clipa, 1)
        gpub = self.core.std.TransferFrame(clipb, 1)

        cpu = self.core.std.Expr(clips=[clipa, clipb], expr=["x 7 + y < x 2 + x 7 - y > x 2 - x 51 * y 49 * + 100 / ? ?", "", ""])
        gpu = self.core.std.Expr(clips=[gpua, gpub], expr=["x 7 + y < x 2 + x 7 - y > x 2 - x 51 * y 49 * + 100 / ? ?", "", ""])

        gpu = self.core.std.TransferFrame(gpu, 0)

        self.checkDifference(cpu, gpu)


if __name__ == '__main__':
    unittest.main()

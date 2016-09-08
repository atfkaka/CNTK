using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Configuration;
using System.Diagnostics;

using Microsoft.MSR.CNTK.Extensibility.Managed;

namespace Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClientTest
{
    public class Program
    {
        private static void Main(string[] args)
        {
            var model = @"\Projects\Kraken\CNTK\en\models\model_1.dnn";
            var data = @"\Projects\Kraken\CNTK\en\Data\cntk_test.txt";

            // Default to auto
            int devideId = -4;

            var argValues = new Dictionary<string, string>();
            if (!ParseCommandLine(args, argValues))
            {
                PrintUsage();
                return;
            }

            model = argValues["M"];
            data = argValues["T"];

            if (argValues.ContainsKey("D") && !int.TryParse(argValues["D"], out devideId))
            {
                PrintUsage();
                return;
            }

            if (argValues.ContainsKey("P"))
            {
                Evaluate(model, 100000, devideId);
            }
            else if (argValues.ContainsKey("E"))
            {
                Evaluate(model, data, devideId);
            }
            else
            {
                Console.WriteLine("parameter not supported.");
            }
        }

        private static bool ParseCommandLine(string[] args, Dictionary<string, string> values)
        {
            HashSet<string> singleParams = new HashSet<string>() { "P", "E" };

            try
            {
                for (int i = 0; i < args.Length; i++)
                {
                    var key = args[i].Trim().ToUpperInvariant();

                    if (!key.StartsWith("/"))
                    {
                        Console.Error.WriteLine("Invalid command: " + key);
                        Console.Error.WriteLine();
                        return false;
                    }

                    key = key.TrimStart('/');
                    if (singleParams.Contains(key))
                    {
                        values.Add(key, null);
                        continue;
                    }

                    var value = args[++i].Trim();

                    values.Add(key, value);
                }

                if (values.ContainsKey("P") && values.ContainsKey("E") ||
                    !values.ContainsKey("P") && !values.ContainsKey("E") ||
                    !values.ContainsKey("M") ||
                    !values.ContainsKey("T"))
                {
                    Console.Error.WriteLine("Missing a required parameter");
                    Console.Error.WriteLine();
                    return false;
                }

            }
            catch (Exception e)
            {
                Console.Error.WriteLine("Error parsing command line: " + e.Message);
                Console.Error.WriteLine();
                return false;
            }

            return true;
        }

        private static void PrintUsage()
        {
            Console.WriteLine("CnktWrapper [/p | /e] [/m modelPath] [/t testDataPath] [/d deviceId]");
            Console.WriteLine("  /p                 Run perf evaluation of model.");
            Console.WriteLine("  /e                 Evaluate model for precision, recall.");
            Console.WriteLine("  /m modelPath       Specifies the CNTK model to evaluate.");
            Console.WriteLine("  /t testDataPath    Specifies the test data file path to run the model with.");
            Console.WriteLine("  /d deviceId        Specifies whether to run on GPU or CPU. Defaults to auto.");
            Console.WriteLine("                     -1 for CPU");
            Console.WriteLine("                     0 for GPU or index of specific GPU.");
            Console.WriteLine();
        }

        private static void Evaluate(string modelPath, float size, int deviceId, bool append = false)
        {
            Stopwatch totalTimer = new Stopwatch();
            totalTimer.Start();
            Stopwatch timer = new Stopwatch();
            List<long> samples = new List<long>();

            var rnd = new Random((int)DateTime.Now.Ticks % Int16.MaxValue);

            var proc = Process.GetCurrentProcess();
            var privateBefore = proc.PrivateMemorySize64;
            var virtualBefore = proc.VirtualMemorySize64;

            using (var model = new IEvaluateModelManagedF())
            {
                Console.WriteLine("Loading CNTK model...");
                //model.Init("parallelTrain=false\r\ndistributedMBReading=true\r\n");
                model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelPath), deviceId: deviceId);

                long privateEvalSum = 0;
                long virtualEvalSum = 0;
                proc.Refresh();
                var privateAfterInit = proc.PrivateMemorySize64 - privateBefore;
                var virtualAfterInit = proc.VirtualMemorySize64 - virtualBefore;
                Console.WriteLine("Init Memory: {0}\t{1}", privateAfterInit.ToMib(), virtualAfterInit.ToMib());

                var inDims = model.GetNodeDimensions(NodeGroup.Input);
                var outDims = model.GetNodeDimensions(NodeGroup.Output);
                int inputSize = inDims["features"];
                Console.WriteLine("Input size: {0}", inputSize);


                Console.WriteLine("Evaluating test set...");

                for (int i = 0; i < size; i++)
                {
                    if (i % 500 == 0)
                    {
                        Console.Write("\r{0}, {1:P4}", i, i / size);
                    }


                    var inputFeatures = Enumerable.Range(1, inputSize).Select(f => (float)rnd.Next(255)).ToList();
                    var features = new Dictionary<string, List<float>>();
                    features.Add(inDims.Keys.First(), inputFeatures);

                    proc.Refresh();
                    var evalPrivateBefore = proc.PrivateMemorySize64;
                    var evalVirtualBefore = proc.VirtualMemorySize64;
                    timer.Restart();

                    model.Evaluate(features, outDims.Keys.First());

                    samples.Add(timer.ElapsedMilliseconds);

                    proc.Refresh();
                    privateEvalSum += proc.PrivateMemorySize64 - evalPrivateBefore;
                    virtualEvalSum += proc.VirtualMemorySize64 - evalVirtualBefore;
                }

                proc.Refresh();
                var privateAfterEval = proc.PrivateMemorySize64 - privateAfterInit;
                var virtualAfterEval = proc.VirtualMemorySize64 - virtualAfterInit;
                var privateTotal = proc.PrivateMemorySize64 - privateBefore;
                var virtualTotal = proc.VirtualMemorySize64 - virtualBefore;

                Console.WriteLine();
                var apm = privateEvalSum / size;
                var avm = virtualEvalSum / size;
                Console.WriteLine("Eval time (μs), avg:{0:N4} stdev:{1:N4} min:{2} max:{3}", samples.Average(), samples.StdDev(), samples.Min(), samples.Max());
                Console.WriteLine("Total eval time: {0} m", totalTimer.Elapsed.TotalMinutes);
                Console.WriteLine("Init Memory: {0}\t{1}", privateAfterInit.ToMib(), virtualAfterInit.ToMib());
                Console.WriteLine("Eval Memory: {0}\t{1}", privateAfterEval.ToMib(), virtualAfterEval.ToMib());
                Console.WriteLine("Total Memory: {0}\t{1}", privateTotal.ToMib(), virtualTotal.ToMib());
                Console.WriteLine("Avg eval memory: {0}\t{1}", apm.ToMib(), avm.ToMib());

                WriteStatsFile(modelPath, append, deviceId, samples, privateAfterInit, virtualAfterInit, privateAfterEval, virtualAfterEval, apm, avm);
            }
        }

        private static void Evaluate(string modelPath, string dataPath, int deviceId, bool append = false)
        {
            Console.WriteLine("Evaluating using devideId: {0}", deviceId);
            Console.WriteLine("Reading test data...");
            var lines = File.ReadAllLines(dataPath);
            Console.WriteLine("Samples: {0}", lines.Length);

            List<long> samples = new List<long>(lines.Length);

            using (var model = new IEvaluateModelManagedF())
            {
                Console.WriteLine("Loading CNTK model...");
                model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelPath), deviceId: deviceId);

                var inDims = model.GetNodeDimensions(NodeGroup.Input);
                var outDims = model.GetNodeDimensions(NodeGroup.Output);
                int inputSize = inDims["features"];
                Console.WriteLine("Input size: {0}", inputSize);
                Console.WriteLine("Output name: {0}, Output size: {1}", outDims.First().Key, outDims.First().Value);

                var data = new List<Vector>();

                Console.WriteLine("Evaluating test set...");

                for (int i = 0; i < lines.Length; i++)
                {
                    if (i % 500 == 0 || i == lines.Length - 1)
                    {
                        Console.Write("\r{0}, {1:P4}", i, i / (float)lines.Length);
                    }

                    var v = LoadData(lines[i], inputSize);

                    if (v.Skipped)
                    {
                        continue;
                    }

                    v.Output = model.Evaluate(v.Features, outDims.First().Key);//"ScaledLogLikelihood");


                    v.OutputLabel = v.Output.IndexOf(v.Output.Max()).ToString()[0];
                    v.Features = null;

                    data.Add(v);
                }

                Console.WriteLine();
                Console.WriteLine("Writing output file...");

                //var resultFile = string.Format("RawResults_{0}.tsv");
                //File.WriteAllLines(resultFile, data.Where(v => !v.Skipped).Select(v => string.Format("{0}\t{1}\t{2}", v.Label, v.OutputLabel, string.Join("\t", v.Output))));

                var testDataFileName = Path.GetFileNameWithoutExtension(dataPath);
                File.WriteAllLines(string.Format("{0}_fp.txt", testDataFileName), data.Where(v => !v.Skipped && v.Label == '0' && v.OutputLabel == '1').Select(x => x.Original));
                File.WriteAllLines(string.Format("{0}_fn.txt", testDataFileName), data.Where(v => !v.Skipped && v.Label == '1' && v.OutputLabel == '0').Select(x => x.Original));

                Console.WriteLine("Computing results...");
                var skipped = data.Count(v => v.Skipped);
                var p = (double)data.Count(v => !v.Skipped && v.Label == '1');
                var n = (double)data.Count(v => !v.Skipped && v.Label == '0');
                var tp = (double)data.Count(v => !v.Skipped && v.Label == '1' && v.OutputLabel == '1');
                var tn = (double)data.Count(v => !v.Skipped && v.Label == '0' && v.OutputLabel == '0');
                var fp = (double)data.Count(v => !v.Skipped && v.Label == '0' && v.OutputLabel == '1');
                var fn = (double)data.Count(v => !v.Skipped && v.Label == '1' && v.OutputLabel == '0');

                double predP = tp + fp;
                double precision = predP == 0 ? 0 : tp / predP;
                double recall = tp + fn == 0 ? 0 : tp / (tp + fn);
                double total = tp + tn + fp + fn;
                double accuracy = total == 0 ? 0 : (tp + tn) / total;
                double error = total == 0 ? 0 : (fp + fn) / total;
                double f1 = 2 * ((precision * recall) / (precision + recall));

                Console.WriteLine("Skipped: {0}", skipped);
                Console.WriteLine("Positive: {0}", p);
                Console.WriteLine("Negative: {0}", n);
                Console.WriteLine();

                Debug.Assert(tp + fn == p);
                Debug.Assert(fp + tn == n);
                double tpr = p == 0 ? 0 : tp / p;
                double fpr = n == 0 ? 0 : fp / n;


                double predN = fn + tn;
                double ppv = predP == 0 ? 0 : tp / predP;   // Positive Predictive Value
                double nor = predN == 0 ? 0 : fn / predN;   // false(Negative) Omission Rate

                Console.WriteLine("{0,10} {1,10} {2,10} {3,10}", "", "P", "N", "Recall");
                Console.WriteLine("{0,10} {1,10:N0} {2,10:N0} {3,10:N4}", "P", tp, fn, tpr);
                Console.WriteLine("{0,10} {1,10:N0} {2,10:N0} {3,10:N4}", "N", fp, tn, fpr);
                Console.WriteLine("{0,10} {1,10:N4} {2,10:N4}", "Precision", ppv, nor);
                Console.WriteLine();

                Console.WriteLine("accuracy:  {0:N6}", accuracy);
                Console.WriteLine("error:     {0:N6}", error);
                Console.WriteLine();

                Console.WriteLine("f1:        {0:N6}", f1);
                Console.WriteLine("precision: {0:N6}", precision);
                Console.WriteLine("recall:    {0:N6}", recall);
                Console.WriteLine();

                //WriteStatsFile(modelPath, append, deviceId, samples, privateAfterInit, virtualAfterInit, privateAfterEval, virtualAfterEval, apm, avm);
            }
        }

        private static void WriteStatsFile(string modelPath, bool append, int deviceId, List<long> samples, long privateAfterInit, long virtualAfterInit, long privateAfterEval, long virtualAfterEval, float apm, float avm)
        {
            var resultsFileName = string.Format("PermMeasurements.tsv");
            bool writeHeader = !File.Exists(resultsFileName);

            var modelName = Path.GetFileName(modelPath);
            File.WriteAllLines(string.Format("Samples_{0}_{1}.tsv", modelName, deviceId == -1 ? "cpu" : "gpu"), samples.Select(x => x.ToString()));

            using (var sw = new StreamWriter(resultsFileName, append))
            {
                //if (writeHeader)
                //{
                //    sw.WriteLine(string.Join("\t", "Model", "DeviceId", "TP", "FP", "TN", "FN", "accuracy", "error", "precision", "recall", "f1", "latency", "PIM", "VIM", "TPM", "TVM", "APM", "AVM"));
                //}

                //var modelName = Path.GetFileName(modelPath);
                //sw.WriteLine(string.Join("\t", modelName, deviceId, tp, fp, tn, fn, accuracy, error, precision, recall, f1,
                //                            latency, privateAfterInit.ToMib(), virtualAfterInit.ToMib(), privateAfterEval.ToMib(), virtualAfterEval.ToMib(), apm.ToMib(), avm.ToMib()));
                if (writeHeader)
                {
                    sw.WriteLine(string.Join("\t", "Model", "DeviceId", "latency:avg", "latency:stdev", "latency:min", "latency:max", "IPM", "IVM", "EPM", "EVM", "APM", "AVM"));
                }

                sw.WriteLine(string.Join("\t", modelName, deviceId, samples.Average(), samples.StdDev(), samples.Min(), samples.Max(),
                                            privateAfterInit.ToMib(), virtualAfterInit.ToMib(), privateAfterEval.ToMib(), virtualAfterEval.ToMib(), apm.ToMib(), avm.ToMib()));
            }
        }

        private static Vector LoadData(string line, int dimensions)
        {
            var vector = new Vector()
            {
                Original = line
            };

            var lineParts = line.Split(new char[] { '|' }, StringSplitOptions.RemoveEmptyEntries);

            if (lineParts[1] == "F ")
            {
                vector.Skipped = true;
                return vector;
            }


            var label = '~';
            if (lineParts[0].Contains(":"))
            {
                //L x:1
                label = lineParts[0][2];
            }
            else
            {
                //L 0 1
                var oneHotLabel = lineParts[0].Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Skip(1).Select(x => int.Parse(x)).ToList();
                label = oneHotLabel.IndexOf(oneHotLabel.Max()).ToString()[0];
            }
            vector.Label = label;
            if (label != '0' && label != '1')
            {
                throw new Exception("invalid label: " + label);
            }

            var featureValues = lineParts[1].Split(' ');
            float[] features = new float[dimensions]; /// Why did I need "+1" ????  + 1];

            foreach (var v in featureValues.Skip(1))
            {
                var parts = v.Split(':');
                features[int.Parse(parts[0])] = float.Parse(parts[1]);
            }

            vector.Features.Add("features", new List<float>(features));

            return vector;
        }
    }

    public static class Extensions
    {
        private static float bytesInMib = 1048576;

        public static float ToMib(this float x)
        {
            return x / bytesInMib;
        }

        public static float ToMib(this long x)
        {
            return x / bytesInMib;
        }

        public static double ToMib(this double x)
        {
            return x / bytesInMib;
        }

        public static double StdDev(this List<long> values)
        {
            if (values.Count == 0)
            {
                return 0;
            }

            double avg = values.Average();
            double sum = values.Sum(d => Math.Pow(d - avg, 2));
            return Math.Sqrt((sum) / (values.Count() - 1));
        }

        public static double Percentile(this List<long> sequence, double percentile)
        {
            sequence.Sort();
            int N = sequence.Count;
            double n = (N - 1) * percentile + 1;
            if (n == 1)
            {
                return sequence[0];
            }
            else if (n == N)
            {
                return sequence[N - 1];
            }

            int k = (int)n;
            double d = n - k;
            return sequence[k - 1] + d * (sequence[k] - sequence[k - 1]);
        }
    }
    public class Vector
    {
        public char Label { get; set; }
        public Dictionary<string, List<float>> Features { get; set; }

        public List<float> Output { get; set; }
        public char OutputLabel { get; set; }
        public bool Skipped { get; set; }

        public string Original { get; set; }
        public Vector()
        {
            Features = new Dictionary<string, List<float>>();
        }
    }
}

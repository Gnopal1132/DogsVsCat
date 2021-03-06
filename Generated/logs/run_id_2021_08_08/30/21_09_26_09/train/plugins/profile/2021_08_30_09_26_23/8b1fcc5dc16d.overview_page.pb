?	?z?юqe@?z?юqe@!?z?юqe@	W??S@W??S@!W??S@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?z?юqe@?XP?	@1??.R(_=@A?|%????Iyͫ:?? @YYk(??`@*	??ʡs A2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?vL??`@!^O?n?X@)?vL??`@1^O?n?X@:Preprocessing2F
Iterator::ModelePmp"?`@!      Y@)+O ????1?$IxkҐ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismc??m?`@!m{H???X@)?
?Ov??1???o??|?:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap???>?`@!ʼ?N?X@)??ܵ?|??1??Z[xx?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 78.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9W??S@I?+?ݟ?@Qe?2?1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?XP?	@?XP?	@!?XP?	@      ??!       "	??.R(_=@??.R(_=@!??.R(_=@*      ??!       2	?|%?????|%????!?|%????:	yͫ:?? @yͫ:?? @!yͫ:?? @B      ??!       J	Yk(??`@Yk(??`@!Yk(??`@R      ??!       Z	Yk(??`@Yk(??`@!Yk(??`@b      ??!       JGPUYW??S@b q?+?ݟ?@ye?2?1@?"-
IteratorGetNext/_1_Send??q?Y???!??q?Y???"8
sequential/conv2d/Relu_FusedConv2D?%??Eǧ?!??b+???"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??t%?`??!	???NM??0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?N?K?0??!??[x????0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterۊ?JGƠ?!6̸a????0"i
=gradient_tape/sequential/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltery??p&???!?e??[???0"k
Agradient_tape/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3??B? ??!kv??g#??"g
<gradient_tape/sequential/conv2d_8/Conv2D/Conv2DBackpropInputConv2DBackpropInput?R?UЗ?!?kI?i???0":
sequential/conv2d_8/Relu_FusedConv2D7ڜ"O??!:9s????"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??9e0ٕ?!?ƚ!_??0Q      Y@Y5؝?h???a???\:?X@q???玪?y8{`5???"?	
host?Your program is HIGHLY input-bound because 78.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 
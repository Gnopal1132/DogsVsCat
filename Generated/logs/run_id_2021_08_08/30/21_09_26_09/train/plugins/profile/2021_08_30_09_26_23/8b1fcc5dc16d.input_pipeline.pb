	?z?юqe@?z?юqe@!?z?юqe@	W??S@W??S@!W??S@"w
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
	?XP?	@?XP?	@!?XP?	@      ??!       "	??.R(_=@??.R(_=@!??.R(_=@*      ??!       2	?|%?????|%????!?|%????:	yͫ:?? @yͫ:?? @!yͫ:?? @B      ??!       J	Yk(??`@Yk(??`@!Yk(??`@R      ??!       Z	Yk(??`@Yk(??`@!Yk(??`@b      ??!       JGPUYW??S@b q?+?ݟ?@ye?2?1@
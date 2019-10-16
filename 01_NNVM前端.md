# NNVM前端(以MXNet到NNVM为例)
一些框架中的算子虽然和nnvm中一些算子功能相同, 但命名不同, 故而需要将框架中的算子转为nnvm中的算子. nnvm中所支持的op在初始化阶段会进行注册, 其注册过程参考[NNVM的中OP注册机制](00_NNVM中的OP注册机制.md).</br>

## 与NNVM中op的初步映射

实际转换的工作主要由`nnvm/python/nnvm/frontend/`中的相关文件实现, 该目录下基本每个文件对应一个框架到NNVM IR的转换.</br>

以mxnet为例, `mxnet.py`中的`from_mxnet()`函数即可将一个mxnet模型转换为NNVM IR. 其中首先调用`_sort_topo()`对mxnet图中所有节点做一次拓扑排序, 并以拓扑序遍历图中所有的node(即op), 对于每个op都执行一次_convert_symbol()函数，以将mxnet中的op转为nnvm中的op. 其核心代码如下:</br>
```python
identity_list = identity_list if identity_list else _identity_list
convert_map = convert_map if convert_map else _convert_map
if op_name in identity_list:    # 不需要Convert, 直接用此op_name
    op = get_nnvm_op(op_name)
    sym = op(*inputs, **attrs)
elif op_name in convert_map:    # 需要修改mxnet的op_name至nnvm的op_name
    sym = convert_map[op_name](inputs, attrs)
return sym
```

`_identity_list`和`_convert_map`即用于存储mxnet中op到nnvm中op的映射.</br>
* `_identity_list`记录那些mxnet中和nnvm名字, 参数一致的op, 若一个mxnet中op在此list内, 则直接调用`get_nnvm_op()`函数通过op_name获取nnvm中的"op". 
* `_convert_map`记录那些mxnet和nnvm中不一致的op. 其是一个`string->func ptr`的映射, 对于每一个mxnet op_name, 都有一个对应的convert函数, 该函数做了一些输入的转换后也是调用`get_nnvm_op()`函数通过op_name获取nnvm中的"op". </br>

## 从nnvm op到symbol

在前端`get_nnvm_op(op_name)`函数中使用getattr(_sym, op_name)返回symbol.py中名为op_name的attr, 但直接查看`nnvm/python/nnvm/symbol.py`并不能找到相关attr, 比如"conv2d"等等, 而op又和symbol有什么关系, 这些可以参考可以参考[NNVM中的OP与Symbol]().</br>

总而言之, 初始化阶段会为每个nnvm中的op在`nnvm.symbol`中创建了一个`nnvm.symbol.op_name(*args, **kwargs)`函数, `get_nnvm_op()`就是获得了此函数. 该函数功能即根据一个op创建一个对应的symbol, 可以看到获取到此函数之后, 系统会通过`sym = op(*inputs, **attrs)`创建一个sym并返回.

## 整体建图过程
上面主要分析了一个mxnet中的op怎样得到其在nnvm中的symbol, 现在看一下更宏观的利用这些nnvm中symbol构图的过程, 其核心代码如下:
```python
# symbol是传进来的mxnet的图, graph初始时是一个空dict
for sym in _topo_sort(symbol):
    ...
    node = _convert_symbol(op_name, [], attr)   #　对于每个op都获得对应的symbol
    graph[name] = node  # convert的这些symbol会被放入字典中
nodes = []
for sym in symbol:  # 对于mxnet中每个结点
    node = get_node(sym)    # get_node()函数是返回这个symbol的output_index
    nodes.append(node)

if len(nodes) > 1:
    return _sym.Group(nodes)# ---- todo
```
完成图的重建之后, from_mxnet()函数将原始的param转换为tvm.ndArray类型, 这部分比较简单不多赘述. 总之, 最终from_mxnet返回以下两个参数:
* `nnvm.Symbol`对象，用于存储网络模型描述
* `tvm.ndArray`对象，用于存储网络参数





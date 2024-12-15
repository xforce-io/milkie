# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import ast
import difflib
import importlib
import math
import random
import typing
import logging
from typing import Any, ClassVar, Dict, List, Optional

from milkie.interpreter.base import BaseInterpreter
from milkie.interpreter.interpreter_error import InterpreterError

logger = logging.getLogger(__name__)

class InternalPythonInterpreter(BaseInterpreter):
    r"""A customized python interpreter to control the execution of
    LLM-generated codes. The interpreter makes sure the code can only execute
    functions given in action space and import white list. It also supports
    fuzzy variable matching to retrieve uncertain input variable name.

    .. highlight:: none

    This class is adapted from the hugging face implementation
    `python_interpreter.py <https://github.com/huggingface/transformers/blob/8f
    093fb799246f7dd9104ff44728da0c53a9f67a/src/transformers/tools/python_interp
    reter.py>`_. The original license applies::

        Copyright 2023 The HuggingFace Inc. team. All rights reserved.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
        implied. See the License for the specific language governing
        permissions and limitations under the License.

    We have modified the original code to suit our requirements. We have
    encapsulated the original functions within a class and saved the
    interpreter state after execution. We have added support for "import"
    statements, "for" statements, and several binary and unary operators. We
    have added import white list to keep `import` statement safe. Additionally,
    we have modified the variable matching logic and introduced the
    :obj:`fuzz_state` for fuzzy matching.

    Modifications copyright (C) 2023 CAMEL-AI.org

    Args:
        action_space (Dict[str, Any], optional): A dictionary that maps action
            names to their corresponding functions or objects. The interpreter
            can only execute functions that are either directly listed in this
            dictionary or are member functions of objects listed in this
            dictionary. The concept of :obj:`action_space` is derived from
            EmbodiedAgent, representing the actions that an agent is capable of
            performing. If `None`, set to empty dict. (default: :obj:`None`)
        import_white_list (List[str], optional): A list that stores
            the Python modules or functions that can be imported in the code.
            All submodules and functions of the modules listed in this list are
            importable. Any other import statements will be rejected. The
            module and its submodule or function name are separated by a period
            (:obj:`.`). (default: :obj:`None`)
        unsafe_mode (bool, optional): If `True`, the interpreter runs the code
            by `eval()` without any security check. (default: :obj:`False`)
        raise_error (bool, optional): Raise error if the interpreter fails.
            (default: :obj:`False`)
    """

    _CODE_TYPES: ClassVar[List[str]] = ["python", "py", "python3", "python2"]

    def __init__(
        self,
        action_space: Optional[Dict[str, Any]] = None,
        import_white_list: Optional[List[str]] = None,
        unsafe_mode: bool = False,
        raise_error: bool = False,
    ) -> None:
        self.action_space = action_space or dict()
        self.state = self.action_space.copy()
        self.fuzz_state: Dict[str, Any] = dict()
        self.import_white_list = import_white_list or list()
        self.raise_error = raise_error
        self.unsafe_mode = unsafe_mode
        
        # 添加内置函数到 state 中
        self._add_builtins_to_state()

    def _add_builtins_to_state(self):
        # 添加常用的内置函数到 state 中
        builtins = {
            'len': len,
            'range': range,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'sum': sum,
            'max': max,
            'min': min,
            'random': random,
            'math': math,
            'print': print,
            "open": open,
            "abs": abs,
            "type": type,
            # 可以根据需要添加更多内置函数
        }
        self.state.update(builtins)

    def run(self, code: str, code_type: str, varDict: Optional[Dict[str, Any]] = None) -> Any:
        r"""Executes the given code with specified code type in the
        interpreter.

        This method takes a string of code and its type, checks if the code
        type is supported, and then executes the code. If `unsafe_mode` is
        set to `False`, the code is executed in a controlled environment using
        the `execute` method. If `unsafe_mode` is `True`, the code is executed
        using `eval()` with the action space as the global context. An
        `InterpreterError` is raised if the code type is unsupported or if any
        runtime error occurs during execution.

        Args:
            code (str): The python code to be executed.
            code_type (str): The type of the code, which should be one of the
            supported code types (`python`, `py`, `python3`, `python2`).


        Returns:
            Any: The output of the executed code.

        Raises:
            InterpreterError: If the `code_type` is not supported or if any
                runtime error occurs during the execution of the code.
        """
        if code_type not in self._CODE_TYPES:
            raise InterpreterError(
                f"Unsupported code type {code_type}. "
                f"`{self.__class__.__name__}` only supports "
                f"{', '.join(self._CODE_TYPES)}."
            )
        if not self.unsafe_mode:
            return self.execute(code, state=varDict)
        else:
            return eval(code, self.action_space, globals=varDict)

    def update_action_space(self, action_space: Dict[str, Any]) -> None:
        r"""Updates action space for *python* interpreter."""
        self.action_space.update(action_space)

    def supported_code_types(self) -> List[str]:
        r"""Provides supported code types by the interpreter."""
        return self._CODE_TYPES

    def execute(
        self,
        code: str,
        state: Optional[Dict[str, Any]] = None,
        fuzz_state: Optional[Dict[str, Any]] = None,
        keep_state: bool = True,
    ) -> Any:
        r"""Execute the input python codes in a security environment.

        Args:
            code (str): Generated python code to be executed.
            state (Optional[Dict[str, Any]], optional): External variables that
                may be used in the generated code. (default: :obj:`None`)
            fuzz_state (Optional[Dict[str, Any]], optional): External variables
                that do not have certain variable names. The interpreter will
                use fuzzy matching to access these variables. For example, if
                :obj:`fuzz_state` has a variable :obj:`image`, the generated
                code can use :obj:`input_image` to access it. (default:
                :obj:`None`)
            keep_state (bool, optional):  If :obj:`True`, :obj:`state` and
                :obj:`fuzz_state` will be kept for later execution. Otherwise,
                they will be cleared. (default: :obj:`True`)

        Returns:
            Any: The value of the last statement (excluding "import") in the
                code. For this interpreter, the value of an expression is its
                value, the value of an "assign" statement is the assigned
                value, and the value of an "if" and "for" block statement is
                the value of the last statement in the block.
        """
        if state is not None:
            self.state.update(state)
        if fuzz_state is not None:
            self.fuzz_state.update(fuzz_state)

        try:
            expression = ast.parse(code)
        except SyntaxError as e:
            if self.raise_error:
                raise InterpreterError(f"Syntax error in code: {e}")
            else:
                import traceback

                return traceback.format_exc()

        result = None
        for idx, node in enumerate(expression.body):
            try:
                line_result = self._execute_ast(node)
            except InterpreterError as e:
                if not keep_state:
                    self.clear_state()
                raise InterpreterError({e})

            if line_result is not None:
                result = line_result

        if not keep_state:
            self.clear_state()

        return result

    def clear_state(self) -> None:
        r"""Initialize :obj:`state` and :obj:`fuzz_state`."""
        self.state = self.action_space.copy()
        self.fuzz_state = {}

    # ast.Index is deprecated after python 3.9, which cannot pass type check,
    # but is still necessary for older versions.
    @typing.no_type_check
    def _execute_ast(self, expression: ast.AST) -> Any:
        if isinstance(expression, ast.Break):
            raise StopIteration  # 使用 StopIteration 来处理 break
        elif isinstance(expression, ast.Continue):
            return None  # continue 语句返回 None
        elif isinstance(expression, ast.Pass):
            return None  # 添加对 pass 语句的支持
        elif isinstance(expression, ast.Assign):
            # Assignment -> evaluate the assignment which should
            # update the state. We return the variable assigned as it may
            # be used to determine the final result.
            return self._execute_assign(expression)
        elif isinstance(expression, ast.Attribute):
            value = self._execute_ast(expression.value)
            return getattr(value, expression.attr)
        elif isinstance(expression, ast.BinOp):
            # Binary Operator -> return the result value
            return self._execute_binop(expression)
        elif isinstance(expression, ast.Call):
            # Function call -> return the value of the function call
            return self._execute_call(expression)
        elif isinstance(expression, ast.Compare):
            # Compare -> return True or False
            return self._execute_condition(expression)
        elif isinstance(expression, ast.Constant):
            # Constant -> just return the value
            return expression.value
        elif isinstance(expression, ast.Dict):
            # Dict -> evaluate all keys and values
            result: Dict = {}
            for k, v in zip(expression.keys, expression.values):
                if k is not None:
                    result[self._execute_ast(k)] = self._execute_ast(v)
                else:
                    result.update(self._execute_ast(v))
            return result
        elif isinstance(expression, ast.Expr):
            # Expression -> evaluate the content
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.For):
            return self._execute_for(expression)
        elif isinstance(expression, ast.FormattedValue):
            # Formatted value (part of f-string) -> evaluate the content
            # and return
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.If):
            # If -> execute the right branch
            return self._execute_if(expression)
        elif isinstance(expression, ast.Import):
            # Import -> add imported names in self.state and return None.
            self._execute_import(expression)
            return None
        elif isinstance(expression, ast.ImportFrom):
            self._execute_import_from(expression)
            return None
        elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
            # cannot pass type check
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.JoinedStr):
            return "".join(
                [str(self._execute_ast(v)) for v in expression.values]
            )
        elif isinstance(expression, ast.List):
            # List -> evaluate all elements
            return [self._execute_ast(elt) for elt in expression.elts]
        elif isinstance(expression, ast.Name):
            # Name -> pick up the value in the state
            return self._execute_name(expression)
        elif isinstance(expression, ast.Subscript):
            # Subscript -> return the value of the indexing
            return self._execute_subscript(expression)
        elif isinstance(expression, ast.Tuple):
            return tuple([self._execute_ast(elt) for elt in expression.elts])
        elif isinstance(expression, ast.UnaryOp):
            # Binary Operator -> return the result value
            return self._execute_unaryop(expression)
        elif isinstance(expression, ast.With):
            return self._execute_with(expression)
        elif isinstance(expression, ast.Try):
            return self._execute_try(expression)
        elif isinstance(expression, ast.FunctionDef):
            return self._execute_function_def(expression)
        elif isinstance(expression, ast.IfExp):
            # 三元表达式 -> 根据条件返回相应的值
            return self._execute_ifexp(expression)
        elif isinstance(expression, ast.AugAssign):
            # 增强赋值操作 -> 执行操作并更新状态
            return self._execute_augassign(expression)
        elif isinstance(expression, ast.ListComp):
            # 列表推导式 -> 执行并返回结果列表
            return self._execute_listcomp(expression)
        elif isinstance(expression, ast.BoolOp):
            # 处理布尔运算符
            values = [self._execute_ast(value) for value in expression.values]
            if isinstance(expression.op, ast.And):
                result = all(values)
            elif isinstance(expression.op, ast.Or):
                result = any(values)
            else:
                raise InterpreterError(f"Unsupported boolean operator: {expression.op}")
            return result
        else:
            # For now we refuse anything else. Let's add things as we need
            # them.
            raise InterpreterError(
                f"{expression.__class__.__name__} is not supported."
            )

    def _execute_assign(self, assign: ast.Assign) -> Any:
        targets = assign.targets
        result = self._execute_ast(assign.value)

        for target in targets:
            self._assign(target, result)
        return result

    def _assign(self, target: ast.expr, value: Any):
        if isinstance(target, ast.Name):
            self.state[target.id] = value
        elif isinstance(target, ast.Tuple):
            if not isinstance(value, tuple):
                raise InterpreterError(
                    f"Expected type tuple, but got "
                    f"{value.__class__.__name__} instead."
                )
            if len(target.elts) != len(value):
                raise InterpreterError(
                    f"Expected {len(target.elts)} values but got "
                    f"{len(value)}."
                )
            for t, v in zip(target.elts, value):
                self._assign(t, v)
        elif isinstance(target, ast.Subscript):
            container = self._execute_ast(target.value)
            index = self._execute_ast(target.slice)
            try:
                container[index] = value
            except TypeError:
                raise InterpreterError(
                    f"Cannot assign to subscript of type "
                    f"{container.__class__.__name__}"
                )
        else:
            raise InterpreterError(
                f"Unsupported variable type. Expected "
                f"ast.Name, ast.Tuple, or ast.Subscript, got "
                f"{target.__class__.__name__} instead."
            )

    def _execute_call(self, call: ast.Call) -> Any:
        callable_func = self._execute_ast(call.func)

        # Todo deal with args
        args = [self._execute_ast(arg) for arg in call.args]
        kwargs = {
            keyword.arg: self._execute_ast(keyword.value)
            for keyword in call.keywords
        }
        return callable_func(*args, **kwargs)

    def _execute_subscript(self, subscript: ast.Subscript):
        """处理下标访问操作"""
        if isinstance(subscript.slice, ast.Slice):
            # 支持切片操作
            value = self._execute_ast(subscript.value)
            start = self._execute_ast(subscript.slice.lower) if subscript.slice.lower else None
            stop = self._execute_ast(subscript.slice.upper) if subscript.slice.upper else None
            step = self._execute_ast(subscript.slice.step) if subscript.slice.step else None
            return value[start:stop:step]
        else:
            # 普通索引访问
            index = self._execute_ast(subscript.slice)
            value = self._execute_ast(subscript.value)
            
            if not isinstance(subscript.ctx, ast.Load):
                raise InterpreterError(
                    f"{subscript.ctx.__class__.__name__} is not supported for "
                    "subscript."
                )
            
            # 处理不同类型的容器
            if isinstance(value, (list, tuple)):
                # 对于列表和元组，直接使用整数索引
                return value[int(index)]
            elif isinstance(value, dict):
                # 对于字典，先检查键是否存在
                if index in value:
                    return value[index]
                # 如果键是字符串，尝试模糊匹配
                if isinstance(index, str):
                    close_matches = difflib.get_close_matches(
                        index,
                        [key for key in list(value.keys()) if isinstance(key, str)],
                    )
                    if close_matches:
                        return value[close_matches[0]]
            elif isinstance(value, str):
                # 对于字符串，确保索引是整数
                if isinstance(index, int):
                    return value[index]
                raise InterpreterError(f"String indices must be integers, not {type(index).__name__}")

            raise InterpreterError(f"Could not index {type(value).__name__} with '{index}'")

    def _execute_name(self, name: ast.Name):
        if isinstance(name.ctx, ast.Store):
            return name.id
        elif isinstance(name.ctx, ast.Load):
            return self._get_value_from_state(name.id)
        else:
            raise InterpreterError(f"{name.ctx} is not supported.")

    def _execute_condition(self, condition: ast.Compare):
        left = self._execute_ast(condition.left)
        
        for op, comparator in zip(condition.ops, condition.comparators):
            right = self._execute_ast(comparator)
            
            if isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.NotEq):
                result = left != right
            elif isinstance(op, ast.Lt):
                result = left < right
            elif isinstance(op, ast.LtE):
                result = left <= right
            elif isinstance(op, ast.Gt):
                result = left > right
            elif isinstance(op, ast.GtE):
                result = left >= right
            elif isinstance(op, ast.Is):
                result = left is right
            elif isinstance(op, ast.IsNot):
                result = left is not right
            elif isinstance(op, ast.In):
                result = left in right
            elif isinstance(op, ast.NotIn):
                result = left not in right
            else:
                raise InterpreterError(f"不支持的操作符: {op}")
            
            if not result:
                return False
            
            left = right  # 为下一次比较准备
        
        return True

    def _execute_if(self, if_statement: ast.If):
        result = None
        if self._execute_ast(if_statement.test):
            for line in if_statement.body:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result
        else:
            for line in if_statement.orelse:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result
        return result

    def _execute_for(self, for_statement: ast.For) -> Any:
        result = None
        try:
            for value in self._execute_ast(for_statement.iter):
                self._assign(for_statement.target, value)
                for line in for_statement.body:
                    try:
                        line_result = self._execute_ast(line)
                        if line_result is not None:
                            result = line_result
                    except StopIteration:
                        return result  # break 语句触发时返回当前结果
        except StopIteration:
            pass  # 处理外层的 break

        # 执行 else 子句（如果有的话）
        if hasattr(for_statement, 'orelse') and for_statement.orelse:
            for line in for_statement.orelse:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result

        return result

    def _execute_import(self, import_module: ast.Import) -> None:
        for module in import_module.names:
            self._validate_import(module.name)
            alias = module.asname or module.name
            self.state[alias] = importlib.import_module(module.name)

    def _execute_import_from(self, import_from: ast.ImportFrom):
        if import_from.module is None:
            raise InterpreterError("\"from . import\" is not supported.")
        for import_name in import_from.names:
            full_name = import_from.module + f".{import_name.name}"
            self._validate_import(full_name)
            imported_module = importlib.import_module(import_from.module)
            alias = import_name.asname or import_name.name
            self.state[alias] = getattr(imported_module, import_name.name)

    def _validate_import(self, full_name: str):
        tmp_name = ""
        found_name = False
        for name in full_name.split("."):
            tmp_name += name if tmp_name == "" else f".{name}"
            if tmp_name in self.import_white_list:
                found_name = True
                return

        if not found_name:
            raise InterpreterError(
                f"It is not permitted to import modules "
                f"than module white list (try to import "
                f"{full_name})."
            )

    def _execute_binop(self, binop: ast.BinOp):
        left = self._execute_ast(binop.left)
        operator = binop.op
        right = self._execute_ast(binop.right)

        if isinstance(operator, ast.Add):
            return left + right
        elif isinstance(operator, ast.Sub):
            return left - right
        elif isinstance(operator, ast.Mult):
            return left * right
        elif isinstance(operator, ast.Div):
            return left / right
        elif isinstance(operator, ast.FloorDiv):
            return left // right
        elif isinstance(operator, ast.Mod):
            return left % right
        elif isinstance(operator, ast.Pow):
            return left**right
        elif isinstance(operator, ast.LShift):
            return left << right
        elif isinstance(operator, ast.RShift):
            return left >> right
        elif isinstance(operator, ast.MatMult):
            return left @ right
        else:
            raise InterpreterError(f"Operator not supported: {operator}")

    def _execute_unaryop(self, unaryop: ast.UnaryOp):
        operand = self._execute_ast(unaryop.operand)
        operator = unaryop.op

        if isinstance(operator, ast.UAdd):
            return +operand
        elif isinstance(operator, ast.USub):
            return -operand
        elif isinstance(operator, ast.Not):
            return not operand
        else:
            raise InterpreterError(f"Operator not supported: {operator}")

    def _execute_with(self, with_stmt: ast.With) -> Any:
        context_managers = []
        for item in with_stmt.items:
            cm = self._execute_ast(item.context_expr)
            if item.optional_vars:
                self._assign(item.optional_vars, cm.__enter__())
            context_managers.append(cm)

        result = None
        try:
            for stmt in with_stmt.body:
                result = self._execute_ast(stmt)
        finally:
            for cm in reversed(context_managers):
                cm.__exit__(None, None, None)

        return result

    def _execute_try(self, try_stmt: ast.Try) -> Any:
        try:
            for stmt in try_stmt.body:
                result = self._execute_ast(stmt)
        except Exception as e:
            for handler in try_stmt.handlers:
                if handler.type is None or isinstance(e, self._execute_ast(handler.type)):
                    if handler.name:
                        self.state[handler.name] = e
                    for stmt in handler.body:
                        result = self._execute_ast(stmt)
                    break
            else:
                raise
        else:
            return result
        finally:
            for stmt in try_stmt.finalbody:
                self._execute_ast(stmt)

    def _execute_function_def(self, func_def: ast.FunctionDef) -> None:
        def function(*args, **kwargs):
            # 创建一个新的局部作用域
            local_scope = {}
            
            # 处理参数
            for arg, value in zip(func_def.args.args, args):
                local_scope[arg.arg] = value
            
            # 处理默认参数
            defaults = func_def.args.defaults
            for arg, default in zip(reversed(func_def.args.args), reversed(defaults)):
                if arg.arg not in local_scope:
                    local_scope[arg.arg] = self._execute_ast(default)
            
            # 处理关键字参数
            for kwarg, value in kwargs.items():
                local_scope[kwarg] = value
            
            # 保存当前状态
            old_state = self.state.copy()
            
            # 更新状态包含局部变量
            self.state.update(local_scope)
            
            result = None
            try:
                # 执行函数体
                for stmt in func_def.body:
                    result = self._execute_ast(stmt)
                    if isinstance(stmt, ast.Return):
                        break
            finally:
                # 恢复原始状态
                self.state = old_state
            
            return result

        # 将函数添加到解释器的状态中
        self.state[func_def.name] = function

    def _execute_return(self, return_stmt: ast.Return) -> Any:
        if return_stmt.value:
            return self._execute_ast(return_stmt.value)
        else:
            return None

    def _get_value_from_state(self, key: str) -> Any:
        if key in self.state:
            return self.state[key]
        elif key in self.fuzz_state:
            return self.fuzz_state[key]
        else:
            # 尝试从 Python 内置函数中获取
            builtin_value = getattr(__builtins__, key, None)
            if builtin_value is not None:
                return builtin_value
            
            # 如果仍然找不到，尝试模糊匹配
            close_matches = difflib.get_close_matches(
                key, list(self.fuzz_state.keys()), n=1
            )
            if close_matches:
                return self.fuzz_state[close_matches[0]]
            else:
                raise InterpreterError(f"The variable `{key}` is not defined.")

    def _execute_ifexp(self, ifexp: ast.IfExp) -> Any:
        condition = self._execute_ast(ifexp.test)
        if condition:
            return self._execute_ast(ifexp.body)
        else:
            return self._execute_ast(ifexp.orelse)

    def _execute_augassign(self, augassign: ast.AugAssign) -> Any:
        target = self._execute_ast(augassign.target)
        value = self._execute_ast(augassign.value)
        op = augassign.op

        if isinstance(op, ast.Add):
            result = target + value
        elif isinstance(op, ast.Sub):
            result = target - value
        elif isinstance(op, ast.Mult):
            result = target * value
        elif isinstance(op, ast.Div):
            result = target / value
        elif isinstance(op, ast.FloorDiv):
            result = target // value
        elif isinstance(op, ast.Mod):
            result = target % value
        elif isinstance(op, ast.Pow):
            result = target ** value
        elif isinstance(op, ast.LShift):
            result = target << value
        elif isinstance(op, ast.RShift):
            result = target >> value
        elif isinstance(op, ast.BitOr):
            result = target | value
        elif isinstance(op, ast.BitXor):
            result = target ^ value
        elif isinstance(op, ast.BitAnd):
            result = target & value
        else:
            raise InterpreterError(f"不支持的增强赋值操作符: {op}")

        self._assign(augassign.target, result)
        return result

    def _execute_listcomp(self, listcomp: ast.ListComp) -> List[Any]:
        result = []
        generators = listcomp.generators
        
        def execute_generators(generators, current_index=0):
            if current_index == len(generators):
                result.append(self._execute_ast(listcomp.elt))
                return

            gen = generators[current_index]
            iterable = self._execute_ast(gen.iter)
            
            for item in iterable:
                self._assign(gen.target, item)
                
                if all(self._execute_ast(if_expr) for if_expr in gen.ifs):
                    execute_generators(generators, current_index + 1)
        
        execute_generators(generators)
        return result

    def _execute_while(self, while_statement: ast.While) -> Any:
        result = None
        try:
            while self._execute_ast(while_statement.test):
                for line in while_statement.body:
                    try:
                        line_result = self._execute_ast(line)
                        if line_result is not None:
                            result = line_result
                    except StopIteration:
                        return result  # break 语句触发时返回当前结果
        except StopIteration:
            pass  # 处理外层的 break

        # 执行 else 子句（如果有的话）
        if hasattr(while_statement, 'orelse') and while_statement.orelse:
            for line in while_statement.orelse:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result

        return result

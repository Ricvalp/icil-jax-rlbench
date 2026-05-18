from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
from flax import traverse_util
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from icil_jax_rlbench.models.direct_regression_policy import DirectRegressionPolicy


class TrainState(train_state.TrainState):
    rng: jax.Array


def tree_global_norm(tree: Any) -> jnp.ndarray:
    leaves = [jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree_util.tree_leaves(tree)]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.stack(leaves)))


def clip_tree_by_global_norm(tree: Any, max_norm: float) -> Tuple[Any, jnp.ndarray]:
    norm = tree_global_norm(tree)
    if max_norm is None or float(max_norm) <= 0.0:
        return tree, norm
    scale = jnp.minimum(1.0, jnp.asarray(float(max_norm), dtype=jnp.float32) / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda x: x * scale.astype(x.dtype), tree), norm


def make_name_mask(params: Any, include: Sequence[str] = (), exclude: Sequence[str] = ()) -> Any:
    include = tuple(str(x) for x in include if str(x))
    exclude = tuple(str(x) for x in exclude if str(x))
    flat = traverse_util.flatten_dict(params, sep='/')
    out = {}
    for path, value in flat.items():
        name = str(path)
        keep = True if not include else any(pat in name for pat in include)
        if exclude and any(pat in name for pat in exclude):
            keep = False
        out[path] = jnp.asarray(keep, dtype=jnp.bool_)
    return traverse_util.unflatten_dict(out, sep='/')


def apply_mask(tree: Any, mask: Optional[Any]) -> Any:
    if mask is None:
        return tree
    return jax.tree_util.tree_map(lambda x, m: jnp.where(m, x, jnp.zeros_like(x)), tree, mask)


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(pred.astype(jnp.float32) - target.astype(jnp.float32)))


def l1_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred.astype(jnp.float32) - target.astype(jnp.float32)))


def action_loss(pred: jnp.ndarray, target: jnp.ndarray, loss_type: str = 'mse') -> jnp.ndarray:
    if loss_type == 'mse':
        return mse_loss(pred, target)
    if loss_type == 'l1':
        return l1_loss(pred, target)
    if loss_type == 'huber':
        return jnp.mean(optax.huber_loss(pred.astype(jnp.float32), target.astype(jnp.float32), delta=1.0))
    raise ValueError(f'Unknown loss_type={loss_type!r}')


def _split_rng(rng: jax.Array, n: int) -> Tuple[jax.Array, jax.Array]:
    rng, out = jax.random.split(rng)
    return rng, out


@dataclass(frozen=True)
class StepConfig:
    loss_type: str = 'mse'
    grad_clip_norm: float = 1.0
    first_order: bool = True
    inner_lr: float = 1e-2
    inner_grad_clip_norm: float = 1.0
    memory_grad_clip_norm: float = 1.0
    memory_update_clip_norm: float = 0.0
    inner_param_include: Tuple[str, ...] = ()
    inner_param_exclude: Tuple[str, ...] = ()
    log_attention_stats: bool = False


def create_pretrain_step(model: DirectRegressionPolicy, cfg: StepConfig) -> Callable[[TrainState, Dict[str, jnp.ndarray]], Tuple[TrainState, Dict[str, jnp.ndarray]]]:
    def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state.rng)
        dropout_rng = jax.random.fold_in(step_rng, jax.lax.axis_index('devices'))

        def loss_fn(params):
            if bool(cfg.log_attention_stats):
                pred, attn_stats = model.apply(
                    {'params': params}, batch, train=True, return_attn_stats=True, rngs={'dropout': dropout_rng}
                )
            else:
                pred = model.apply({'params': params}, batch, train=True, rngs={'dropout': dropout_rng})
                attn_stats = {}
            loss = action_loss(pred, batch['target_action'], cfg.loss_type)
            return loss, {'loss': loss, 'pred_l1': l1_loss(pred, batch['target_action']), **attn_stats}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads, grad_norm = clip_tree_by_global_norm(grads, cfg.grad_clip_norm)
        grads = jax.lax.pmean(grads, axis_name='devices')
        metrics = jax.lax.pmean({**metrics, 'grad_norm': grad_norm}, axis_name='devices')
        state = state.apply_gradients(grads=grads).replace(rng=rng)
        return state, metrics

    return jax.pmap(train_step, axis_name='devices')


def _task_param_outer_loss(
    model: DirectRegressionPolicy,
    params: Any,
    task_batch: Dict[str, jnp.ndarray],
    *,
    loss_type: str,
    train: bool,
    rng: jax.Array,
    log_attention_stats: bool = False,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    if bool(log_attention_stats):
        pred, attn_stats = model.apply(
            {'params': params}, task_batch, train=train, return_attn_stats=True, rngs={'dropout': rng}
        )
    else:
        pred = model.apply({'params': params}, task_batch, train=train, rngs={'dropout': rng})
        attn_stats = {}
    loss = action_loss(pred, task_batch['target_action'], loss_type)
    return loss, {'pred_l1': l1_loss(pred, task_batch['target_action']), **attn_stats}


def create_param_maml_step(
    model: DirectRegressionPolicy,
    cfg: StepConfig,
    inner_mask: Optional[Any] = None,
) -> Callable[[TrainState, Dict[str, Dict[str, jnp.ndarray]]], Tuple[TrainState, Dict[str, jnp.ndarray]]]:
    def train_step(state: TrainState, batch: Dict[str, Dict[str, jnp.ndarray]]) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state.rng)
        dropout_rng = jax.random.fold_in(step_rng, jax.lax.axis_index('devices'))
        B = batch['query']['target_action'].shape[0]
        task_rngs = jax.random.split(dropout_rng, B)

        def adapt_one(params, task_inner, task_query, task_rng):
            def inner_loss_fn(p, step_batch):
                pred = model.apply({'params': p}, step_batch, train=True, rngs={'dropout': task_rng})
                return action_loss(pred, step_batch['target_action'], cfg.loss_type)

            def body(p, step_batch):
                loss, grad = jax.value_and_grad(inner_loss_fn)(p, step_batch)
                grad = apply_mask(grad, inner_mask)
                grad, grad_norm = clip_tree_by_global_norm(grad, cfg.inner_grad_clip_norm)
                if bool(cfg.first_order):
                    grad = jax.tree_util.tree_map(jax.lax.stop_gradient, grad)
                p_next = optax.apply_updates(p, jax.tree_util.tree_map(lambda g: -float(cfg.inner_lr) * g, grad))
                return p_next, {'inner_loss': loss, 'inner_grad_norm': grad_norm}

            if task_inner:
                adapted, inner_metrics = jax.lax.scan(body, params, task_inner)
                inner_loss = jnp.mean(inner_metrics['inner_loss'])
                inner_grad_norm = jnp.mean(inner_metrics['inner_grad_norm'])
            else:
                adapted = params
                inner_loss = jnp.asarray(0.0, dtype=jnp.float32)
                inner_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
            before, _ = _task_param_outer_loss(model, params, task_query, loss_type=cfg.loss_type, train=True, rng=task_rng)
            after, out_metrics = _task_param_outer_loss(
                model,
                adapted,
                task_query,
                loss_type=cfg.loss_type,
                train=True,
                rng=task_rng,
                log_attention_stats=bool(cfg.log_attention_stats),
            )
            return after, {
                'outer_loss_before': before,
                'outer_loss_after': after,
                'inner_loss': inner_loss,
                'inner_grad_norm': inner_grad_norm,
                'pred_l1': out_metrics['pred_l1'],
                'improvement': before - after,
                **{k: v for k, v in out_metrics.items() if k.startswith('attn_')},
            }

        def loss_fn(params):
            losses, metrics = jax.vmap(adapt_one, in_axes=(None, 0, 0, 0))(params, batch.get('inner', {}), batch['query'], task_rngs)
            loss = jnp.mean(losses)
            metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
            metrics['loss'] = loss
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads, grad_norm = clip_tree_by_global_norm(grads, cfg.grad_clip_norm)
        grads = jax.lax.pmean(grads, axis_name='devices')
        metrics = jax.lax.pmean({**metrics, 'grad_norm': grad_norm}, axis_name='devices')
        state = state.apply_gradients(grads=grads).replace(rng=rng)
        return state, metrics

    return jax.pmap(train_step, axis_name='devices')


def _with_batch_dim(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: v[None, ...] for k, v in batch.items()}


def _tile_memory(memory: jnp.ndarray, memory_mask: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.broadcast_to(memory[None, :, :], (int(n), memory.shape[0], memory.shape[1])), jnp.broadcast_to(memory_mask[None, :], (int(n), memory_mask.shape[0]))


def create_memory_maml_step(model: DirectRegressionPolicy, cfg: StepConfig) -> Callable[[TrainState, Dict[str, Dict[str, jnp.ndarray]]], Tuple[TrainState, Dict[str, jnp.ndarray]]]:
    def train_step(state: TrainState, batch: Dict[str, Dict[str, jnp.ndarray]]) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state.rng)
        dropout_rng = jax.random.fold_in(step_rng, jax.lax.axis_index('devices'))
        B = batch['query']['target_action'].shape[0]
        task_rngs = jax.random.split(dropout_rng, B)

        def predict_with_memory(params, memory, memory_mask, query_batch, task_rng, *, log_attention_stats: bool = False):
            n = int(query_batch['target_action'].shape[0])
            mem_b, mask_b = _tile_memory(memory, memory_mask, n)
            if bool(log_attention_stats):
                pred, attn_stats = model.apply(
                    {'params': params},
                    query_batch,
                    mem_b,
                    support_mask=mask_b,
                    train=True,
                    method=DirectRegressionPolicy.predict_with_memory_and_stats,
                    rngs={'dropout': task_rng},
                )
                return pred, attn_stats
            pred = model.apply(
                {'params': params},
                query_batch,
                mem_b,
                support_mask=mask_b,
                train=True,
                method=DirectRegressionPolicy.predict_with_memory,
                rngs={'dropout': task_rng},
            )
            return pred, {}

        def one_task(params, memory_init_batch, task_inner, task_query, task_rng):
            if bool(cfg.log_attention_stats):
                mem_b, mem_mask_b, support_attn_stats = model.apply(
                    {'params': params},
                    _with_batch_dim(memory_init_batch),
                    train=True,
                    method=DirectRegressionPolicy.encode_support_with_stats,
                    rngs={'dropout': task_rng},
                )
            else:
                mem_b, mem_mask_b = model.apply(
                    {'params': params},
                    _with_batch_dim(memory_init_batch),
                    train=True,
                    method=DirectRegressionPolicy.encode_support,
                    rngs={'dropout': task_rng},
                )
                support_attn_stats = {}
            memory = mem_b[0]
            memory_mask = mem_mask_b[0]
            initial_memory = memory

            def inner_loss_fn(mem, step_batch):
                pred, _ = predict_with_memory(params, mem, memory_mask, step_batch, task_rng)
                return action_loss(pred, step_batch['target_action'], cfg.loss_type)

            def body(mem, step_batch):
                loss, grad = jax.value_and_grad(inner_loss_fn)(mem, step_batch)
                grad, grad_norm = clip_tree_by_global_norm(grad, cfg.memory_grad_clip_norm)
                if bool(cfg.first_order):
                    grad = jax.lax.stop_gradient(grad)
                update = -float(cfg.inner_lr) * grad
                if float(cfg.memory_update_clip_norm) > 0.0:
                    update, _ = clip_tree_by_global_norm(update, cfg.memory_update_clip_norm)
                return mem + update, {'inner_loss': loss, 'inner_grad_norm': grad_norm}

            if task_inner:
                adapted_memory, inner_metrics = jax.lax.scan(body, memory, task_inner)
                inner_loss = jnp.mean(inner_metrics['inner_loss'])
                inner_grad_norm = jnp.mean(inner_metrics['inner_grad_norm'])
            else:
                adapted_memory = memory
                inner_loss = jnp.asarray(0.0, dtype=jnp.float32)
                inner_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
            pred_before, _ = predict_with_memory(params, initial_memory, memory_mask, task_query, task_rng)
            pred_after, attn_stats = predict_with_memory(
                params, adapted_memory, memory_mask, task_query, task_rng, log_attention_stats=bool(cfg.log_attention_stats)
            )
            before = action_loss(pred_before, task_query['target_action'], cfg.loss_type)
            after = action_loss(pred_after, task_query['target_action'], cfg.loss_type)
            mem_delta = tree_global_norm(adapted_memory - initial_memory)
            rel_mem_delta = mem_delta / (tree_global_norm(initial_memory) + 1e-6)
            return after, {
                'outer_loss_before': before,
                'outer_loss_after': after,
                'inner_loss': inner_loss,
                'inner_grad_norm': inner_grad_norm,
                'pred_l1': l1_loss(pred_after, task_query['target_action']),
                'improvement': before - after,
                'memory_delta_norm': mem_delta,
                'memory_relative_delta_norm': rel_mem_delta,
                **support_attn_stats,
                **attn_stats,
            }

        def loss_fn(params):
            losses, metrics = jax.vmap(one_task, in_axes=(None, 0, 0, 0, 0))(params, batch['memory_init'], batch.get('inner', {}), batch['query'], task_rngs)
            loss = jnp.mean(losses)
            metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
            metrics['loss'] = loss
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads, grad_norm = clip_tree_by_global_norm(grads, cfg.grad_clip_norm)
        grads = jax.lax.pmean(grads, axis_name='devices')
        metrics = jax.lax.pmean({**metrics, 'grad_norm': grad_norm}, axis_name='devices')
        state = state.apply_gradients(grads=grads).replace(rng=rng)
        return state, metrics

    return jax.pmap(train_step, axis_name='devices')

import itertools
import os
import lasagne

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.algos.embedding_theano_par.theano_atari import AtariEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rein.algos.embedding_theano_par.trpo_plus_lsh import ParallelTRPOPlusLSH
from rllab.envs.env_spec import EnvSpec
from rllab.spaces.box import Box
from rllab.core.network import ConvNetwork
from sandbox.adam.parallel.gaussian_conv_baseline import ParallelGaussianConvBaseline
from sandbox.haoran.parallel_trpo.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer

stub(globals())

n_seq_frames = 4
n_parallel = 20
model_batch_size = 32
exp_prefix = 'trpo-i-auto-pro-a'
seeds = [0, 1, 2]
etas = [0.001]
mdps = [  # AtariEnv(game='freeway', obs_type="ram+image", frame_skip=4),
    # AtariEnv(game='breakout', obs_type="ram+image", frame_skip=4),
    # AtariEnv(game='frostbite', obs_type="image", frame_skip=4),
    # AtariEnv(game='montezuma_revenge', obs_type="image", frame_skip=4)]
    AtariEnv(game='venture', obs_type="image", frame_skip=4)]
trpo_batch_size = 100000
max_path_length = 4500
dropout = False
batch_norm = True

param_cart_product = itertools.product(
    mdps, etas, seeds
)

for mdp, eta, seed in param_cart_product:
    # mdp_spec = EnvSpec(
    #     observation_space=Box(low=-1, high=1, shape=(1, 128)),
    #     action_space=mdp.spec.action_space
    # )
    #
    # policy = CategoricalMLPPolicy(
    #     env_spec=mdp_spec,
    #     hidden_sizes=(32, 32),
    # )
    # baseline = ParallelLinearFeatureBaseline(env_spec=mdp_spec)

    mdp_spec = EnvSpec(
        observation_space=Box(low=-1, high=1, shape=(1, 52, 52)),
        action_space=mdp.spec.action_space
    )

    network = ConvNetwork(
        input_shape=(n_seq_frames,) + (
            mdp_spec.observation_space.shape[1], mdp_spec.observation_space.shape[2]),
        output_dim=mdp.spec.action_space.flat_dim,
        hidden_sizes=(256,),
        conv_filters=(16, 32),
        conv_filter_sizes=(8, 4),
        conv_strides=(4, 2),
        conv_pads=(0, 0),
    )
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        num_seq_inputs=n_seq_frames,
        prob_network=network,
    )

    env_spec = EnvSpec(
        observation_space=Box(low=-1, high=1, shape=(n_seq_frames, 52, 52)),
        action_space=mdp.spec.action_space
    )

    nips_dqn_args = dict(
        conv_filters=[16, 32],
        conv_filter_sizes=[8, 4],
        conv_strides=[4, 2],
        conv_pads=[(0, 0)] * 2,
        hidden_sizes=[256],
        hidden_nonlinearity=lasagne.nonlinearities.rectify,
    )
    baseline = ParallelGaussianConvBaseline(
        env_spec=env_spec,
        regressor_args=dict(
            optimizer=ParallelConjugateGradientOptimizer(
                subsample_factor=0.1,
                cg_iters=10,
                name="vf_opt",
            ),
            use_trust_region=True,
            step_size=0.01,
            batchsize=trpo_batch_size * 10,
            normalize_inputs=True,
            normalize_outputs=True,
            **nips_dqn_args
        )
    )

    # Alex' settings
    policy_opt_args = dict(
        cg_iters=10,
        reg_coeff=1e-3,
        subsample_factor=0.1,
        max_backtracks=15,
        backtrack_ratio=0.8,
        accept_violation=False,
        hvp_approach=None,
        num_slices=1,
    )

    #
    # model_args = dict(
    #     state_dim=mdp_spec.observation_space.shape,
    #     action_dim=(env_spec.action_space.flat_dim,),
    #     reward_dim=(1,),
    #     layers_disc=[
    #         dict(name='convolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(0, 0),
    #              batch_norm=batch_norm,
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='convolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(1, 1),
    #              batch_norm=batch_norm,
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='convolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(2, 2),
    #              batch_norm=batch_norm,
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='convolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(2, 2),
    #              batch_norm=batch_norm,
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='reshape',
    #              shape=([0], -1)),
    #         dict(name='gaussian',
    #              n_units=1024,
    #              matrix_variate_gaussian=False,
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              batch_norm=batch_norm,
    #              dropout=dropout,
    #              deterministic=True),
    #         dict(name='discrete_embedding',
    #              n_units=128,
    #              batch_norm=batch_norm,
    #              deterministic=True),
    #         dict(name='gaussian',
    #              n_units=1024,
    #              matrix_variate_gaussian=False,
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              batch_norm=batch_norm,
    #              dropout=dropout,
    #              deterministic=True),
    #         dict(name='gaussian',
    #              n_units=1024,
    #              matrix_variate_gaussian=False,
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              batch_norm=batch_norm,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='reshape',
    #              shape=([0], 64, 4, 4)),
    #         dict(name='deconvolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(2, 2),
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              batch_norm=batch_norm,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='deconvolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(1, 1),
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              batch_norm=batch_norm,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='deconvolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(0, 0),
    #              nonlinearity=lasagne.nonlinearities.rectify,
    #              batch_norm=batch_norm,
    #              dropout=False,
    #              deterministic=True),
    #         dict(name='deconvolution',
    #              n_filters=64,
    #              filter_size=(6, 6),
    #              stride=(2, 2),
    #              pad=(0, 0),
    #              nonlinearity=lasagne.nonlinearities.linear,
    #              batch_norm=True,
    #              dropout=False,
    #              deterministic=True),
    #     ],
    #     n_batches=1,
    #     trans_func=lasagne.nonlinearities.rectify,
    #     out_func=lasagne.nonlinearities.linear,
    #     batch_size=model_batch_size,
    #     n_samples=1,
    #     num_train_samples=1,
    #     prior_sd=0.05,
    #     second_order_update=False,
    #     learning_rate=0.0003,
    #     surprise_type=None,
    #     update_prior=False,
    #     update_likelihood_sd=False,
    #     output_type='classfication',
    #     num_classes=32,
    #     likelihood_sd_init=0.1,
    #     disable_variance=False,
    #     ind_softmax=True,
    #     num_seq_inputs=1,
    #     label_smoothing=0.003,
    #     # Disable prediction of rewards and intake of actions, act as actual autoenc
    #     disable_act_rew_paths=True,
    #     # --
    #     # Count settings
    #     # Put penalty for being at 0.5 in sigmoid postactivations.
    #     binary_penalty=10,
    # )

    model_args = dict(
        state_dim=mdp_spec.observation_space.shape,
        action_dim=(env_spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='convolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(1, 1),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='convolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='reshape',
                 shape=([0], -1)),
            dict(name='gaussian',
                 n_units=1024,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=True),
            dict(name='discrete_embedding',
                 n_units=448,
                 batch_norm=batch_norm,
                 deterministic=True),
            dict(name='gaussian',
                 n_units=1024,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=True),
            dict(name='gaussian',
                 n_units=1600,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='reshape',
                 shape=([0], 64, 5, 5)),
            dict(name='deconvolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='deconvolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='deconvolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.linear,
                 batch_norm=True,
                 dropout=False,
                 deterministic=True),
        ],
        n_batches=1,
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=model_batch_size,
        n_samples=1,
        num_train_samples=1,
        prior_sd=0.05,
        second_order_update=False,
        learning_rate=0.0003,
        surprise_type=None,
        update_prior=False,
        update_likelihood_sd=False,
        output_type='classfication',
        num_classes=64,
        likelihood_sd_init=0.1,
        disable_variance=False,
        ind_softmax=True,
        num_seq_inputs=1,
        label_smoothing=0.003,
        # Disable prediction of rewards and intake of actions, act as actual autoenc
        disable_act_rew_paths=True,
        # --
        # Count settings
        # Put penalty for being at 0.5 in sigmoid postactivations.
        binary_penalty=0.1,
    )

    algo = ParallelTRPOPlusLSH(
        n_parallel=n_parallel,
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=trpo_batch_size,
        max_path_length=max_path_length,
        n_itr=1000,
        step_size=0.01,
        optimizer_args=policy_opt_args,
        n_seq_frames=n_seq_frames,
        # --
        # Count settings
        model_pool_args=dict(
            size=5000000,
            min_size=model_batch_size,
            batch_size=model_batch_size,
            subsample_factor=1,
            fill_before_subsampling=False,
        ),
        eta=eta,
        train_model=True,
        train_model_freq=3,
        continuous_embedding=False,
        model_embedding=True,
        sim_hash_args=dict(
            dim_key=32,
            bucket_sizes=None,
            disable_rnd_proj=True,
        ),
        clip_rewards=True,
        model_args=model_args,
    )

    print("Remember, GPUs are linked to seeds!")
    # Note: you have to make sure that imports in all of the imported modules are at their correct location,
    # not the top location. Otherwise, GPU compilation of the AE will fail.
    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        n_parallel=n_parallel,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=False,
        script="sandbox/rein/algos/embedding_theano_par/run_experiment_lite.py",
        # Sync ever 1h.
        periodic_sync_interval=60 * 60,
        sync_all_data_node_to_s3=True
    )
